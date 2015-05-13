# SAEProject2
Software Architecture and Engineering Project 2

TODOS:
------
- [ ] Fix assertions!
- [ ] Implement all the analyze_expr cases -> Attention: With the binaryOps we have to merge states somewhere, right?
- [x] If path is unreachable solver just gives an error: "model is not available", we should just remove the path in this case...
- [x] Implement function calls! (-> note that there are no global variables, so we have only pure functions)
- [ ] write tons of test-cases (according to pdf one for each possible analysis szenarios...)
- [ ] handle assertions in a correct way! -> maybe use z3's assertion solver instead of our own?
- [ ] throw exception if path doesn't return

OPTIMIZATIONS:
--------------
- [x] If Pathconstraint contains a false somewhere, we can direclty throw away that state. Currently we keep it until the end and even give it to the SMT solver!
- [ ] After the Analyzer finished we could just fill in the inputs into the symstore and get the result, but then we would need to store the result variables somewhere. Currently we get the inputs from the z3 solver and then run once again over the hole code with the given interpreter.
