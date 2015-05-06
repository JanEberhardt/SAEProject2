import ast
import numbers

#are we allowed to do this?
import copy
from z3 import *

class SymbolicEngine:
    def __init__(self, function_name, program_ast):
        self.fnc = find_function(program_ast, function_name)
        self.program_ast = program_ast

    # TODO: implement symbolic execution
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
        for finalState in finalStates:
            finalState.solve()
            inputs = finalState.inputs
            fe = FunctionEvaluator(self.fnc, self.program_ast, finalState.inputs)   
            ret = fe.eval()
            input_to_ret.append((finalState.inputs, ret))

        ### End stuff we added


        #TODO What about assertions????
        assetion_violations_to_input = {}

        return (input_to_ret, assetion_violations_to_input)


############
# Analyzer #
############

# This function now returns not only a list of states but also a value 
# -> the evaluated value of the expresssion!
def analyze_expr(expr, state):
    print "    -> analyze_expr()"

    #TODO
    if type(expr) == ast.Tuple:
        r = []
        for el in expr.elts:
            r.append(run_expr(el, fnc))
        return tuple(r)

    if type(expr) == ast.Name:
        #Theres nothing left to do here!
        #if expr.id == 'True':
        #    return 1
        #elif expr.id == 'False':
        #    return 0
        #return state.getSym(expr.id)
        ret = []
        assert(not state.returned)
        fnc = FunctionEvaluator(None, state.ast_root, state.symstore)
        ret.append( (state, run_expr(expr, fnc)) )
        return ret 

    if type(expr) == ast.Num:
        #Theres nothing left to do here!
        #assert (isinstance(expr.n, numbers.Integral))
        #return expr.n
        ret = []
        fnc = FunctionEvaluator(None, state.ast_root, state.symstore)
        ret.append( (state, run_expr(expr, fnc)) )
        return ret 

    #TODO: Totally fine as long as each expr just returns one state
    #Just do it on the crossproduct man!
    if type(expr) == ast.BinOp:
        if type(expr.op) == ast.Add:
            return analyze_expr(expr.left, state) + analyze_expr(expr.right, state)
        if type(expr.op) == ast.Sub:
            return analyze_expr(expr.left, state) - analyze_expr(expr.right, state)
        if type(expr.op) == ast.Mult:
            return analyze_expr(expr.left, state) * analyze_expr(expr.right, state)
        if type(expr.op) == ast.Div:
            return analyze_expr(expr.left, state) / analyze_expr(expr.right, state)
        if type(expr.op) == ast.Mod:
            return analyze_expr(expr.left, state) % analyze_expr(expr.right, state)
        if type(expr.op) == ast.Pow:
            return analyze_expr(expr.left, state) ** analyze_expr(expr.right, state)

        # Evaluate only with constants
        if type(expr.op) == ast.LShift and type(expr.left) == ast.Num and type(expr.right) == ast.Num:
            return analyze_expr(expr.left, state) << analyze_expr(expr.right, state)
        if type(expr.op) == ast.RShift and type(expr.left) == ast.Num and type(expr.right) == ast.Num:
            return analyze_expr(expr.left, state) >> analyze_expr(expr.right, state)

    if type(expr) == ast.UnaryOp:
        if type(expr.op) == ast.Not:
            retValStates = analyze_expr(expr.operand, state)
            for index, tp in enumerate(retValStates):
                left, right = tp
                retValStates[index] = (left, Not(right))
            return retValStates

        if type(expr.op) == ast.USub:
            return -analyze_expr(expr.operand, state)

    if type(expr) == ast.Compare:
        assert (len(expr.ops) == 1)  # Do not allow for x==y==0 syntax
        assert (len(expr.comparators) == 1)
        leftStateVals = analyze_expr(expr.left, state)
        #left_states = analyze_expr(expr.left, state)
        op = expr.ops[0]
        rightStateVals = analyze_expr(expr.comparators[0], state)
        #right_states = analyze_expr(expr.comparators[0], state)

        # Now that we got the states, lets figure out the actual value of the expression
        # in each case...
        #left_values = []
        #for left_state in left_states:
        #    fnc = FunctionEvaluator(None, left_state.ast_root, left_state.symstore)
        #    left_values.append(run_expr(expr.left, fnc))
        #right_values = []
        #for right_state in right_states:
        #    fnc = FunctionEvaluator(None, left_state.ast_root, left_state.symstore)
        #    right_values.append(run_expr(expr.comparators[0], fnc))
        #assert(len(left_values) == len(left_states))
        #assert(len(right_values) == len(right_states))
        
        # Let's build the states that we want to return...
        #TODO this could somewhere be done generically, since we will use this again 
        # and again...
        retValStates = []
        for i in range(0, len(leftStateVals)):
            for j in range(0, len(rightStateVals)):
                e1 = leftStateVals[i][1]
                e2 = rightStateVals[j][1]
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
                #TODO: Merge left and right state!
                newState = state.copy()
                #newState.addConstr(pconstr)
                retValStates.append((newState, ret))
        return retValStates

    #TODO
    if type(expr) == ast.BoolOp:
        if type(expr.op) == ast.And:
            r = True
            for v in expr.values:
                r = r and analyze_expr(v, state)
            return r
        if type(expr.op) == ast.Or:
            r = False
            for v in expr.values:
                r = r or analyze_expr(v, state)
            return r

    #TODO
    if type(expr) == ast.Call:
        f = find_function(state.ast_root, expr.func.id)

        inputs = {}
        assert (len(expr.args) == len(f.args.args))
        # Evaluates all function arguments
        for i in range(0, len(expr.args)):
            inputs[f.args.args[i].id] = analyze_expr(expr.args[i], state)
        
        fnc_a = FunctionAnalyzer(f, state.ast_root, inputs)
        fnc_a.analyze()
        returnStates = []
        for analyzerReturn in fnc_a.analyzerStates:
            returnStates.append(state.mergeWithState(analyzerReturn))
        #TODO: This expression now returns multiple states!
        #TODO: How to handle the return stuff???

    raise Exception('Unhandled expression: ' + ast.dump(expr))

def analyze_stmt(stmt, state):
    print "  -> analyze_stmt()" 

    if type(stmt) == ast.Return:
        returnStates = []
        # This contains a list of states and a list of return values!
        statesVals = analyze_expr(stmt.value, state)
        for stateVal in statesVals:
            assert(not stateVal[0].returned)
            tempState = stateVal[0]
            tempState.return_val = stateVal[1]
            tempState.returned = True
            returnStates.append(stateVal[0])
        return returnStates

    #TODO
    if type(stmt) == ast.If:
        returnStates = []
        #True case
        trueStateVals = analyze_expr(stmt.test, state)
        for trueStateVal in trueStateVals:
            tempState = trueStateVal[0]
            tempState.addConstr(trueStateVal[1])
            returnStates.extend(analyze_body(stmt.body, tempState))  
        #False case
        false_test_ast = ast.UnaryOp()
        false_test_ast.op = ast.Not()
        false_test_ast.operand = stmt.test
        falseStateVals = analyze_expr(false_test_ast, state)
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

        #TODO
        if type(lhs) == ast.Tuple:
            assert (type(rhs) == tuple)
            assert (len(rhs) == len(lhs.elts))
            for el_index in range(len(lhs.elts)):
                el = lhs.elts[el_index]
                assert (type(el) == ast.Name)
                fnc.state[el.id] = rhs[el_index]
            return
        # Standard Case
        if type(lhs) == ast.Name:
            for stateVal in stateVals:
                tempState = stateVal[0]
                fnc = FunctionEvaluator(None, tempState.ast_root, tempState.symstore)
                tempState.addSym(lhs.id, run_expr(stmt.value, fnc))
                returnStates.append(tempState) 
            return returnStates
        
    if type(stmt) == ast.Assert:
        # TODO: implement check whether the assertion holds. 
        # However do not throw exception in case the assertion does not hold.
        # Instead return inputs that trigger the violation from SymbolicEngine.explore()
        return

    raise Exception('Unhandled statement: ' + ast.dump(stmt))

def analyze_body(body, state):
    print "-> anaylze_body()"
    states = []
    states.append(state)
    for stmt in body:
        newStates = []
        for tmpState in states:
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
        # TODO: implement check whether the assertion holds. 
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
        #TODO: Isn't it bad to comment out stuff that was kind of in the reference
        # solution????!!!!!!!!!!!!!!!!!
        #assert (type(f) == ast.FunctionDef)
        #for arg in f.args.args:
        #    assert arg.id in inputs

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
        print "FunctionAnalyzer.analyze()"
        initialState = AnalyzerState(self.ast_root)
        initialState.inputs = self.mainFunctionInputs.copy()
        for key in self.mainFunctionInputs:
            initialState.addSym(key, Int(key))
        print "Initial Symstore: "+str(initialState.symstore)
        self.analyzerStates = analyze_body(self.f.body, initialState)
        for state in self.analyzerStates:
            assert (state.returned)
        print "function '"+self.f.name+"' has "+str(len(self.analyzerStates))+" final state(s):"
        for state in self.analyzerStates:
            state.printMe()
        # Only return the States that did return (in the actual function)
        returnStates = []
        for tmpState in self.analyzerStates:
            if tmpState.returned:
                returnStates.append(tmpState)
        return returnStates

class AnalyzerState:
    def __init__(self, ast_root):

        self.inputs = {}

        # Dictionary that contains variables to symbolics mapping
        self.symstore = {}

        # List of boolean constraints
        self.pconstrs = []

        self.returned = False
        self.returnValue = None
        self.solved = False
        self.solver = Solver()
        self.ast_root = ast_root 
        
    def addSym(self, var, sym):
        self.symstore[var] = sym
        print "         updated or added something in the symstore..."
        print "         current state.symstore: " + str(self.symstore)

    def getSym(self, var):
        return self.symstore[var]
    
    def addConstr(self, constr):
        self.pconstrs.append(constr)
        print "         updated or added something to the pconstrs..."
        print "         current state.pconstrs: " + str(self.pconstrs)

    def printMe(self):
        print "State:"
        print "  symstore: "+str(self.symstore)+" , pconstrs: "+str(self.pconstrs) 

    def copy(self):
        newState = AnalyzerState(self.ast_root)
        newState.inputs = self.inputs.copy()
        newState.symstore = self.symstore.copy()
        newState.pconstrs = copy.copy(self.pconstrs)
        return newState
    
    def mergeWithState(self, otherState):
        mergedState = self.copy()
        for sym in otherState.symstore:
            mergedState.addSym(sym, otherState.getSym(sym))
        for pc in otherState.pconstrs:
            mergedState.addConstr(pc)
        return mergedState

    def solve(self):
        assert (self.returned)
        for pconstr in self.pconstrs:
            self.solver.add(pconstr)
        try:
            self.solver.check()
            model = self.solver.model()
            for key in self.inputs:
                self.inputs[key] = int(str(model[self.symstore[key]]))
            self.solved = True 
        except Z3Exception as ex:
            #TODO: Is there a better way to do this than converting to str?
            if str(ex) == "model is not available":
                print "Tried to solve model but failed, formula contains contradictions, this state will be skipped..."
            else:
                print "Unknown Z3Exception: " + str(ex)

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
