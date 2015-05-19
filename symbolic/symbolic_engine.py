import ast
import numbers

#are we allowed to do this?
import sys
import copy
from z3 import *

class SymbolicEngine:
    def __init__(self, function_name, program_ast):
        self.fnc = find_function(program_ast, function_name)
        self.program_ast = program_ast

    # The return value is a list of tuples [(input#1, ret#1), ...]
    # where input is a dictionary specifying concrete value used as input, e.g. {'x': 1, 'y': 2}
    # and ret is expected return value of the function
    # Note: all returned inputs should explore different program paths
    def explore(self):
        # Generates default input
        input = generate_inputs(self.fnc, {})
        #f = FunctionEvaluator(self.fnc, self.program_ast, input)
        #ret = f.eval()


        input_to_ret = []
        #input_to_ret.append((input, ret))
        
        ### Stuff we added

        # Start the Function Analyzer with the main function
        fa = FunctionAnalyzer(self.fnc, self.program_ast, input)
        finalStates = fa.analyze()
        # Now all the final states are in a list and we just need to give the list
        # to the SMT solver to get the inputs, and then run the evaluator on those
        # inputs...

        assetion_violations_to_input = {}
        for finalState in finalStates:
            assetion_violations_to_input.update(finalState.violated_assertions)

        for finalState in finalStates:
            finalState.solve()
            if not finalState.solved:
                continue
            inputs = finalState.inputs
            ret = finalState.returnValue
            # In case we returned something weird before and the return thing still
            # contains some symbolic stuff run over the program once again with the 
            # evaluator
            if not isinstance(ret, int):
                fe = FunctionEvaluator(self.fnc, self.program_ast, finalState.inputs)   
                ret = fe.eval()
            input_to_ret.append((finalState.inputs, ret))
        print >> sys.stderr, ""
        print >> sys.stderr, ""
        ### End stuff we added

        return (input_to_ret, assetion_violations_to_input)


############
# Analyzer #
############

# This function now returns not only a list of states but also a value 
# -> the evaluated value of the expresssion!
def analyze_expr(expr, state):

    if type(expr) == ast.Tuple:
        # Collect a list of lists of all possible states...
        listOfLists = []
        for el in expr.elts:
            listOfLists.append(analyze_expr(el, state))
        # Take the crossproduct of it and merge the states...
        crossProducts = crossProduct(listOfLists)
        retValStates = []
        for states in crossProducts:
            tempState = state.copy()
            r = []
            for i in range(len(states)):
                tempState = tempState.mergeWithState(states[i][0].copy())
                r.append(states[i][1])
            retValStates.append((tempState, tuple(r)))
        return retValStates

    if type(expr) == ast.Name:
        if expr.id == 'True':
            return [(state, True)]
        elif expr.id == 'False':
            return [(state, False)]
        else:
            return [(state, state.symstore[expr.id])]

    if type(expr) == ast.Num:
        assert (isinstance(expr.n, numbers.Integral))
        return [(state, expr.n)] 

    if type(expr) == ast.BinOp:
        leftStateVals = analyze_expr(expr.left, state)
        rightStateVals = analyze_expr(expr.right, state)
        retValStates = []
        # Doing cross-product here
        for i in range(0, len(leftStateVals)):
            for j in range(0, len(rightStateVals)):
                e1 = leftStateVals[i][1]
                e2 = rightStateVals[j][1]

                if type(expr.op) == ast.Add:
                    ret = e1 + e2
                if type(expr.op) == ast.Sub:
                    ret = e1 - e2
                if type(expr.op) == ast.Mult:
                    ret = e1 * e2
                if type(expr.op) == ast.Div:
                    ret = e1 / e2
                if type(expr.op) == ast.Mod:
                    ret = e1 % e2
                if type(expr.op) == ast.Pow:
                    ret = e1 ** e2

                # Evaluate only with constants
                if type(expr.op) == ast.LShift and type(expr.left) == ast.Num and type(expr.right) == ast.Num:
                    ret = e1 << e2
                if type(expr.op) == ast.RShift and type(expr.left) == ast.Num and type(expr.right) == ast.Num:
                    ret = e1 >> e2

                newState = state.copy()
                newState = newState.mergeWithState(leftStateVals[i][0].copy())
                newState = newState.mergeWithState(rightStateVals[j][0].copy())
                retValStates.append((newState, ret))
        return retValStates

    if type(expr) == ast.UnaryOp:
        if type(expr.op) == ast.Not:
            retValStates = analyze_expr(expr.operand, state)
            for index, tp in enumerate(retValStates):
                state, val = tp
                retValStates[index] = (state, Not(val))
            return retValStates
        if type(expr.op) == ast.USub:
            retValStates = analyze_expr(expr.operand, state)
            for index, tp in enumerate(retValStates):
                state, val = tp
                retValStates[index] = (state, -val)
            return retValStates

    if type(expr) == ast.Compare:
        assert (len(expr.ops) == 1)  # Do not allow for x==y==0 syntax
        assert (len(expr.comparators) == 1)
        leftStateVals = analyze_expr(expr.left, state)
        op = expr.ops[0]
        rightStateVals = analyze_expr(expr.comparators[0], state)
        
        # Let's build the states that we want to return...
        # Doing the cross-product
        retValStates = []
        for i in range(0, len(leftStateVals)):
            for j in range(0, len(rightStateVals)):
                e1 = leftStateVals[i][1]
                e2 = rightStateVals[j][1]

                #Special case for tuples:
                if type(e1) == tuple or type(e2) == tuple:
                    #Case where not the same length or whatever...
                    if type(e1) != type(e2):
                        retValStates.append([state.copy(), False])
                        continue
                    if len(e1) != len(e2):
                        retValStates.append([state.copy(), False])
                        continue

                    #Case where both tuples have same length and we need to compare them...
                    tupleCompState = state.copy()
                    tupleCompState.mergeWithState(leftStateVals[i][0])
                    tupleCompState.mergeWithState(rightStateVals[j][0])
                    e1_ast = ast.parse(str(e1))
                    e2_ast = ast.parse(str(e2))
                    retValStates.extend(tupleComparator(e1_ast.body[0], e2_ast.body[0], tupleCompState))
                    continue

                if type(op) == ast.Eq:
                    ret = (e1 == e2)
                if type(op) == ast.NotEq:
                    ret =  (e1 != e2)
                if type(op) == ast.Gt:
                    ret = (e1 > e2)
                if type(op) == ast.GtE:
                    ret = (e1 >= e2)
                if type(op) == ast.Lt:
                    ret = (e1 < e2)
                if type(op) == ast.LtE:
                    ret = (e1 <= e2)
                newState = state.copy()
                newState = newState.mergeWithState(leftStateVals[i][0].copy())
                newState = newState.mergeWithState(rightStateVals[j][0].copy())
                retValStates.append((newState, ret))
        return retValStates

    if type(expr) == ast.BoolOp:
        if (type(expr.op) == ast.And) or (type(expr.op) == ast.Or):
            expr_stateVals = []
            # Collect a list of lists of all possible states...
            for v in expr.values:
                expr_stateVals.append(analyze_expr(v, state))
            # Take the crossproduct of it
            cpList = crossProduct(expr_stateVals)
            retStateVals = []
            # For each result of the crossproduct merge the states and evaluate...
            for lst in cpList:
                # State in which to carry out the evaluation
                tempState = state.copy()
                if type(expr.op) == ast.And:
                    val = True
                else:
                    val = False
                for stateVal in lst:
                    tempState = tempState.mergeWithState(stateVal[0].copy())
                    if type(expr.op) == ast.And:
                        val = And(val, stateVal[1])
                    else:
                        val = Or(val, stateVal[1])
                retStateVals.append((tempState, val))
            return retStateVals

    if type(expr) == ast.Call:
        f = find_function(state.ast_root, expr.func.id)

        assert (len(expr.args) == len(f.args.args))
        
        #First we need to analyze all the function arguments
        inputsStateValsDict = {}
        for i in range(0, len(expr.args)):
            # Since an expression can return multiple states (function call) we need to
            # again loop here -> and afterwards doing the crossproduct of all the states...
            key = f.args.args[i].id
            inputsStateValsDict[key] = []
            stateVals = analyze_expr(expr.args[i], state)
            for stateVal in stateVals:
                inputsStateValsDict[key].append(stateVal)

        # Now that we have all the states of the function arguments let's 
        # figure out the actual input values!
        
        # Let's build a list with all the crossproducts of the inputs
        listOfLists = []
        for key in inputsStateValsDict:
            listOfLists.append(inputsStateValsDict[key])
        cpList = crossProduct(listOfLists)
        keys = inputsStateValsDict.keys()

        # Build the actual inputs
        inputsList = []
        for lst in cpList:
            tempInput = {}
            assert(len(keys) == len(lst))
            for i in range(len(keys)):
                tempInput[keys[i]] = lst[i][1]
            inputsList.append(tempInput)

        # Run the analyzer on each possible input combination an store 
        # all the possible resulting states in a list
        finalStates = []
        for inputs in inputsList:
            fnc_a = FunctionAnalyzer(f, state.ast_root, inputs)
            finalStates.extend(fnc_a.analyze())

        # Go over the resulting states and merge the contrains wich we need with 'state' 
        retValStates = []
        for finalState in finalStates:
            # Get only the constraints that contain the input variableables:
            constraintsToAdd = finalState.getConstrsOfVars(f.args.args)
            temp = state.copy()
            for constr in constraintsToAdd:
                temp.addConstr(constr)
            retValStates.append((temp, finalState.returnValue))    
        return retValStates

    raise Exception('Unhandled expression: ' + ast.dump(expr))

def analyze_stmt(stmt, state):
    if type(stmt) == ast.Return:
        returnStates = []
        # This contains a list of states and a list of return values!
        statesVals = analyze_expr(stmt.value, state)
        for stateVal in statesVals:
            assert(not stateVal[0].returned)
            tempState = stateVal[0]
            tempState.returnValue = stateVal[1]
            tempState.returned = True
            returnStates.append(stateVal[0])
        return returnStates

    if type(stmt) == ast.If:
        returnStates = []
        #True case
        trueStateVals = analyze_expr(stmt.test, state.copy())
        for trueStateVal in trueStateVals:
            tempState = trueStateVal[0]
            tempState.addConstr(trueStateVal[1])
            returnStates.extend(analyze_body(stmt.body, tempState))  
        #False case
        false_test_ast = ast.UnaryOp()
        false_test_ast.op = ast.Not()
        false_test_ast.operand = stmt.test
        falseStateVals = analyze_expr(false_test_ast, state.copy())
        for falseStateVal in falseStateVals:
            tempState = falseStateVal[0]
            tempState.addConstr(falseStateVal[1])
            returnStates.extend(analyze_body(stmt.orelse, tempState))  
        return returnStates 

    if type(stmt) == ast.Assign:
        assert (len(stmt.targets) == 1)  # Disallow a=b=c syntax
        lhs = stmt.targets[0]
        stateVals = analyze_expr(stmt.value, state)
        returnStates = []
        rhs = analyze_expr(stmt.value, state)

        if type(lhs) == ast.Tuple:
            returnStates = []
            for stateVal in stateVals:
                tempState = stateVal[0]
                for el_index in range(len(lhs.elts)): 
                    el = lhs.elts[el_index]
                    tempState.addSym(el.id, stateVal[1][el_index])
                returnStates.append(tempState) 
            return returnStates

        # Standard Case
        if type(lhs) == ast.Name:
            for stateVal in stateVals:
                tempState = stateVal[0]
                tempState.addSym(lhs.id, stateVal[1])
                returnStates.append(tempState) 
            return returnStates
        
    if type(stmt) == ast.Assert:
        # Implement check whether the assertion holds. 
        # However do not throw exception in case the assertion does not hold.
        # Instead return inputs that trigger the violation from SymbolicEngine.explore()

        # negate expression
        negation = ast.UnaryOp()
        negation.op = ast.Not()
        negation.operand = stmt.test

        returnStates = []
        stateVals = analyze_expr(stmt.test, state)
        for tempState, tempVal in stateVals:
            # Trivial case where we not even need to call the z3 solver...
            if type(tempVal) is bool and tempVal == True:
                returnStates.append(tempState)
                continue
            assertState = tempState.copy()
            assertState.addConstr(Not(tempVal))
            assertState.returned = True
            assertState.solve()
            assertState.printMe() 
            
            # Other trivial case
            if type(tempVal) is bool and tempVal == False:
                tempState.violated_assertions[stmt] = assertState.inputs
                tempState.addConstr(tempVal)
                tempState.returned = True
                returnStates.append(tempState)
                continue

            # Standard case...
            if assertState.solved:
                tempState.violated_assertions[stmt] = assertState.inputs
                print >> sys.stderr, "Found the following violating inputs for assertion: "+str(assertState.inputs)
            else:
                print >> sys.stderr, "Found no inputs for assertion -> means assertion holds in this case..."
            tempState.addConstr(tempVal)
            returnStates.append(tempState)
        return returnStates 

    raise Exception('Unhandled statement: ' + ast.dump(stmt))

def analyze_body(body, state):
    states = [state]
    for stmt in body:
        newStates = []
        for tmpState in states:
            if tmpState.returned:
                newStates.append(tmpState)
            # Here I directly throw away all the states that contain contradictions in 
            # their pconstr!
            elif tmpState.hasContradictions:
                pass
            else:
                newStates.extend(analyze_stmt(stmt, tmpState))
        states = newStates
    return states

###############
# Interpreter #
###############

def run_expr(expr, fnc):
    if type(expr) == ast.Tuple:
        r = []
        for el in expr.elts:
            r.append(run_expr(el, fnc))
        return tuple(r)

    if type(expr) == ast.Name:
        # We changed this one because it makes no sense to return 1 if something is true
        if expr.id == 'True':
            return 1
        elif expr.id == 'False':
            return 0
        return fnc.state[expr.id]

    if type(expr) == ast.Num:
        assert (isinstance(expr.n, numbers.Integral))
        return expr.n

    if type(expr) == ast.BinOp:
        if type(expr.op) == ast.Add:
            return run_expr(expr.left, fnc) + run_expr(expr.right, fnc)
        if type(expr.op) == ast.Sub:
            return run_expr(expr.left, fnc) - run_expr(expr.right, fnc)
        if type(expr.op) == ast.Mult:
            return run_expr(expr.left, fnc) * run_expr(expr.right, fnc)
        if type(expr.op) == ast.Div:
            return run_expr(expr.left, fnc) / run_expr(expr.right, fnc)
        if type(expr.op) == ast.Mod:
            return run_expr(expr.left, fnc) % run_expr(expr.right, fnc)
        if type(expr.op) == ast.Pow:
            return run_expr(expr.left, fnc) ** run_expr(expr.right, fnc)

        # Evaluate only with constants
        if type(expr.op) == ast.LShift and type(expr.left) == ast.Num and type(expr.right) == ast.Num:
            return run_expr(expr.left, fnc) << run_expr(expr.right, fnc)
        if type(expr.op) == ast.RShift and type(expr.left) == ast.Num and type(expr.right) == ast.Num:
            return run_expr(expr.left, fnc) >> run_expr(expr.right, fnc)

    if type(expr) == ast.UnaryOp:
        if type(expr.op) == ast.Not:
            return not run_expr(expr.operand, fnc)
        if type(expr.op) == ast.USub:
            return -run_expr(expr.operand, fnc)

    if type(expr) == ast.Compare:
        assert (len(expr.ops) == 1)  # Do not allow for x==y==0 syntax
        assert (len(expr.comparators) == 1)
        e1 = run_expr(expr.left, fnc)
        op = expr.ops[0]
        e2 = run_expr(expr.comparators[0], fnc)
        if type(op) == ast.Eq:
            return e1 == e2
        if type(op) == ast.NotEq:
            return e1 != e2
        if type(op) == ast.Gt:
            return e1 > e2
        if type(op) == ast.GtE:
            return e1 >= e2
        if type(op) == ast.Lt:
            return e1 < e2
        if type(op) == ast.LtE:
            return e1 <= e2

    if type(expr) == ast.BoolOp:
        if type(expr.op) == ast.And:
            r = True
            for v in expr.values:
                r = r and run_expr(v, fnc)
            return r
        if type(expr.op) == ast.Or:
            r = False
            for v in expr.values:
                r = r or run_expr(v, fnc)
            return r

    if type(expr) == ast.Call:
        f = find_function(fnc.ast_root, expr.func.id)

        inputs = {}
        assert (len(expr.args) == len(f.args.args))
        # Evaluates all function arguments
        for i in range(0, len(expr.args)):
            inputs[f.args.args[i].id] = run_expr(expr.args[i], fnc)

        fnc_eval = FunctionEvaluator(f, fnc.ast_root, inputs)
        return fnc_eval.eval()

    raise Exception('Unhandled expression: ' + ast.dump(expr))


def run_stmt(stmt, fnc):
    if type(stmt) == ast.Return:
        fnc.returned = True
        fnc.return_val = run_expr(stmt.value, fnc)
        return

    if type(stmt) == ast.If:
        cond = run_expr(stmt.test, fnc)
        if cond:
            run_body(stmt.body, fnc)
        else:
            run_body(stmt.orelse, fnc)
        return

    if type(stmt) == ast.Assign:
        assert (len(stmt.targets) == 1)  # Disallow a=b=c syntax
        lhs = stmt.targets[0]
        rhs = run_expr(stmt.value, fnc)
        if type(lhs) == ast.Tuple:
            assert (type(rhs) == tuple)
            assert (len(rhs) == len(lhs.elts))
            for el_index in range(len(lhs.elts)):
                el = lhs.elts[el_index]
                assert (type(el) == ast.Name)
                fnc.state[el.id] = rhs[el_index]
            return
        if type(lhs) == ast.Name:
            fnc.state[lhs.id] = rhs
            return
        
    if type(stmt) == ast.Assert:
        # However do not throw exception in case the assertion does not hold.
        # Instead return inputs that trigger the violation from SymbolicEngine.explore()
        return

    raise Exception('Unhandled statement: ' + ast.dump(stmt))


def run_body(body, fnc):
    for stmt in body:
        run_stmt(stmt, fnc)
        if fnc.returned:
            return

class FunctionEvaluator:
    def __init__(self, f, ast_root, inputs):
        assert (type(f) == ast.FunctionDef)
        for arg in f.args.args:
            assert arg.id in inputs

        self.state = inputs.copy()
        self.returned = False
        self.return_val = None
        self.ast_root = ast_root
        self.f = f

    def eval(self):
        run_body(self.f.body, self)
        assert (self.returned)
        return self.return_val

class FunctionAnalyzer:
    def __init__(self, f, ast_root, inputs):
        self.mainFunctionInputs = inputs
        self.analyzerStates = []
        self.ast_root = ast_root
        self.f = f

    def analyze(self):
        print >> sys.stderr, ""
        print >> sys.stderr, "FunctionAnalyzer.analyze(function:"+str(self.f.name)+")"

        initialState = AnalyzerState(self.ast_root)
        initialState.inputs = self.mainFunctionInputs.copy()

        if self.f.name == 'main':
            for key in self.mainFunctionInputs:
                initialState.addSym(key, Int(key))
        else:
            for key in self.mainFunctionInputs:
                initialState.addSym(key, self.mainFunctionInputs[key])

        self.analyzerStates = analyze_body(self.f.body, initialState)

        print >> sys.stderr, "\nfunction '"+self.f.name+"' has "+str(len(self.analyzerStates))+" final state(s):"

        for state in self.analyzerStates:
            state.printMe()
            print >> sys.stderr, "  returnValue: "+str(state.returnValue)

        # Only return the States that did return (in the actual function)
        returnStates = []
        for tmpState in self.analyzerStates:
            returnStates.append(tmpState)
        return returnStates

class AnalyzerState:
    def __init__(self, ast_root):

        self.inputs = {}

        # Dictionary that contains variables to symbolics mapping
        self.symstore = {}

        # List of boolean constraints
        self.pconstrs = []

        # List of violated assertions
        self.violated_assertions = {}

        self.returned = False
        self.returnValue = None
        self.hasContradictions = False
        self.solved = False
        self.solver = Solver()
        self.ast_root = ast_root 
        
    def addSym(self, var, sym):
        self.symstore[var] = sym

    def getSym(self, var):
        return self.symstore[var]
    
    def addConstr(self, constr):
        # Check if 'constr' is not trivial (like true false)
        if type(constr) is bool:
            if constr == True:
                return
            elif constr == False:
                self.hasContradictions = True
                return
        if constr.eq(Not(True)):
            self.hasContradictions = True
            return
        if constr.eq(Not(False)):
            return
        # Check if this constraint doesn't exist yet:
        for pconst in self.pconstrs:
            if constr.eq(pconst):
                return
        self.pconstrs.append(constr)

    def printMe(self):
        print >> sys.stderr, "State:"
        print >> sys.stderr, "  symstore: "+str(self.symstore)+" , pconstrs: "+str(self.pconstrs)+" returnValue: "+str(self.returnValue)

    def copy(self):
        newState = AnalyzerState(self.ast_root)
        newState.inputs = self.inputs.copy()
        newState.symstore = self.symstore.copy()
        newState.pconstrs = copy.copy(self.pconstrs)
        newState.returnValue = copy.copy(self.returnValue)
        newState.hasContradictions = copy.copy(self.hasContradictions)
        return newState
    
    def mergeWithState(self, otherState):
        mergedState = self.copy()
        for sym in otherState.symstore:
            assert(self.getSym(sym) == otherState.getSym(sym))
        for pc in otherState.pconstrs:
            mergedState.addConstr(pc)
        return mergedState

    def getConstrsOfVars(self, variables):
        constraints = []
        for pconst in self.pconstrs:
            assert not (type(pconst) is bool)
            if containsAllElements(variables, get_vars(pconst)):
                constraints.append(pconst)
        return constraints

    def solve(self):
        if not self.returned:
            raise Exception("There's a path that doesn't return")
        for pconstr in self.pconstrs:
            self.solver.add(pconstr)
        if str(self.solver.check()) == "sat" and not self.hasContradictions:
            model = self.solver.model()
            print >> sys.stderr, "SMT solver model: " + str(model)
            for key in model:
                self.inputs[str(key)] = int(str(model[key]))
            self.solved = True 
        else:
            print >> sys.stderr, "   found model that contains contradictions -> cannot be solved..."
            

####################
# Helper Functions #
####################

# f: function for which to generate inputs
# inputs: dictionary that maps argument names to values. e.g. {'x': 42 }
def generate_inputs(f, inputs):
    inputs = {}
    for arg in f.args.args:
        assert (type(arg) == ast.Name)
        if arg.id in inputs:
            inputs[arg.id] = inputs[arg.id]
        else:
            # By default input are set to zero
            inputs[arg.id] = 0

    return inputs


def find_function(p, function_name):
    assert (type(p) == ast.Module)
    for x in p.body:
        if type(x) == ast.FunctionDef and x.name == function_name:
            return x
    raise LookupError('Function %s not found' % function_name)

# Extracts variables used from a Z3 expression
# Taken from http://z3.codeplex.com/SourceControl/changeset/view/fbce8160252d#src/api/python/z3util.py
def get_vars(f):
    r = set()
    def collect(f):
        if is_const(f):
            if f.decl().kind() == Z3_OP_UNINTERPRETED and not askey(f) in r:
                r.add(askey(f))
        else:
            for c in f.children():
                collect(c)
    collect(f)
    return [str(var.n) for var in r]

# Wrapper for allowing Z3 ASTs to be stored into Python Hashtables.
class AstRefKey:
    def __init__(self, n):
        self.n = n
    def __hash__(self):
        return self.n.hash()
    def __eq__(self, other):
        return self.n.eq(other.n)
    def __repr__(self):
        return str(self.n)

def askey(n):
    assert isinstance(n, AstRef)
    return AstRefKey(n)

def containsAllElements(list1, list2): 
    # Returns True if first list contiains every element of second list
    for l2 in list2:
        if l2 in list1:
            continue
        else:
            return True
    return True 

# becomes as an argument a list of lists and returns all possible combinations:
# ex: crossProduct([[a,b,c],[d,e]]) returns [[a,d], [a,e], [b,d], [b,e], [c,d],[c,d]]
def crossProduct(listOfLists):
    if len(listOfLists)<2:
        return listOfLists

    temp =[]
    for t in listOfLists[0]:
        temp.append([t])
    for i in range(1, len(listOfLists)):
        temp = crossProductHelper(temp, listOfLists[i])
    return temp

def crossProductHelper(left, right):
    result = []
    for i in range(len(left)):
        for j in range(len(right)):
            temp = copy.copy(left[i])
            temp.append(copy.copy(right[j]))
            result.append(temp)
    return result

#Compares two tuples in a symbolic way... 
def tupleComparator(left, right, state):
    retStateVals = []
    monsterAndRoot = ast.BoolOp()
    monsterAndRoot.op = ast.And()
    tt = ast.Name()
    tt.id = 'True'
    monsterAndRoot.values = [tt]
    assert(len(left.value.elts) == len(right.value.elts))
    assert(len(left.value.elts)>0)
    for i in range(len(left.value.elts)):
        compare_ast = ast.Compare()
        compare_ast.ops = [ast.Eq()]
        compare_ast.left = left.value.elts[i]
        compare_ast.comparators = [right.value.elts[i]]
        compare_subAnd = ast.BoolOp()
        compare_subAnd.op = ast.And()
        compare_subAnd.values = [compare_ast]
        if len(monsterAndRoot.values) == 1:
            compare_subAnd.values.append(monsterAndRoot.values[0])
            monsterAndRoot = compare_subAnd
        else:
            temp = copy.copy(monsterAndRoot)
            compare_subAnd.values.append(temp)
            monsterAndRoot = compare_subAnd

    ret = analyze_expr(monsterAndRoot, state)
    return ret


