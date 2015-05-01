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
        #print input
        #f = FunctionEvaluator(self.fnc, self.program_ast, input)
        #ret = f.eval()

        input_to_ret = []
        #input_to_ret.append((input, ret))
        
        ### Stuff we added
        f = FunctionAnalyzer(self.fnc, self.program_ast, input)
        finalStates = f.analyze()
        
        # Now all the analyzer states are in the anSts list and we can start to build
        # the formulas and give them to the SMT solver.TODO
        # Each AnalyzerState then results in a different Input.

        # Once we have the Inputs we can get the return values.
        # To get the needed return values of our inputs we need to run something like:
        #for anSt in finalStates:
            #input = generate_inputs(self.fnc, inputs)
            #f = FunctionEvaluator(self.fnc, self.program_ast, input)
            #ret = f.eval()
            #input_to_ret.append((input, ret))
        ### End stuff we added


        #TODO What about assertions????
        assetion_violations_to_input = {}

        return (input_to_ret, assetion_violations_to_input)


############
# Analyzer #
############

def analyze_expr(expr, state):
    print "    -> analyze_expr()"

    #TODO
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
        return state.getSym(expr.id)

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
            return not analyze_expr(expr.operand, state)
        if type(expr.op) == ast.USub:
            return -analyze_expr(expr.operand, state)

    if type(expr) == ast.Compare:
        assert (len(expr.ops) == 1)  # Do not allow for x==y==0 syntax
        assert (len(expr.comparators) == 1)
        e1 = analyze_expr(expr.left, state)
        op = expr.ops[0]
        e2 = analyze_expr(expr.comparators[0], state)
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
        
        fnc_a = FunctionAnalyzer(f, fnc.ast_root, inputs)
        return fnc_a.analyze()

    raise Exception('Unhandled expression: ' + ast.dump(expr))

def analyze_stmt(stmt, state):
    print "  -> analyze_stmt()" 

    if type(stmt) == ast.Return:
        state.returned = True
        state.return_val = analyze_expr(stmt.value, state)
        returnStates = []
        returnStates.append(state)
        assert (returnStates[0].returned)
        return returnStates

    #TODO
    if type(stmt) == ast.If:
        #analyze ture_condition:
        true_cond = analyze_expr(stmt.test, state)
        #create the opposit condition by inserting a new node in the ast!
        notExpr = ast.UnaryOp()
        notExpr.op = ast.Not()
        notExpr.operand = stmt.test

        #analyze false condition
        false_cond = analyze_expr(notExpr, state)

        state_true = state
        state_false = state.copy()
        state_true.addConstr(true_cond)
        state_false.addConstr(false_cond)

        returnStates = []
        returnStatesTrue = analyze_body(stmt.body, state_true)
        returnStatesFalse = analyze_body(stmt.orelse, state_false)
        returnStates = returnStatesTrue + returnStatesFalse
        return returnStates

    if type(stmt) == ast.Assign:
        assert (len(stmt.targets) == 1)  # Disallow a=b=c syntax
        lhs = stmt.targets[0]
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
            state.addSym(lhs.id, rhs)
            returnStates = []
            returnStates.append(state)
            #print "assigment is now returning "+str(len(returnStates))+" state(s)"
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
    print "run_expr()"
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
    print "run_stmt()" 
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
        assert (type(f) == ast.FunctionDef)
        for arg in f.args.args:
            assert arg.id in inputs

        self.state = inputs.copy()
        self.returned = False
        self.return_val = None
        self.ast_root = ast_root
        self.f = f

    def eval(self):
        print "FunctionEvaluator.eval()"
        run_body(self.f.body, self)
        print "functionEvaluator.state: "+str(state)

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
        initialState = AnalyzerState()
        for key in self.mainFunctionInputs:
            initialState.addSym(key, Int(key))
        print "Initial Symstore: "+str(initialState.symstore)
        self.analyzerStates = analyze_body(self.f.body, initialState)
        for state in self.analyzerStates:
            assert (state.returned)
        print "function x has "+str(len(self.analyzerStates))+" final state(s):"
        for state in self.analyzerStates:
            print "symstore: "+str(state.symstore)+" pconstr: "+str(state.pconstrs)
        # Only return the States that did return (in the actual function)
        returnStates = []
        for tmpState in self.analyzerStates:
            if tmpState.returned:
                returnStates.append(tmpState)
        return returnStates

class AnalyzerState:
    def __init__(self):

        # Dictionary that contains variables to symbolics mapping
        self.symstore = {}

        # List of boolean constraints
        self.pconstrs = []

        # Boolean that says wheter the function returned in this state
        # TODO: Is this necessary? Can there be cases where we didn't return
        # at a leaf?
        self.returned = False
        self.return_val = None
        
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

    def copy(self):
        newState = AnalyzerState()
        #TODO: How to copy the symstore???
        #newState.symstore = copy.deepcopy(self.symstore)
        newState.symstore = self.symstore.copy()
        newState.pconstrs = copy.deepcopy(self.pconstrs)
        return newState

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
