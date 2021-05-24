# Disassembly-Sequence-Planner
Non-deterministic algorithm which adapts to removal failure to 'predict' uncertainty, outputting a disassembly string
Based on the work described, the following limitations exist:
1.	The use of probability ranges as training data allows for unfeasible sequences to occur, adding noise to our results. 
2.	The stochastic training data provides a degree of randomness, thus, contributing to a non-deterministic output.
3.	After a significant number of cycles the search mechanism is unlikely to escape the local minima.
4.	The Algorithm only considers initial conditions, passes and failures. Thus, some bizarre removal decisions concerning geometry, timing and tooling remain viable.
5.	The Algorithm is being evaluated as an analytical tool and as such steady state conditions are realised. For real time use, mean aggregate confidence will not be fed back into the initial conditions, instead confidence will be maintained cycle to cycle, allowing the algorithm to remain agile.
6.	Increases in product complexity will quickly render the Algorithm unsuitable for DSP generation due to its’ iterative nature.
7.  A great deal of matrix pre-processing is required for new products
