## AZsPC - My answer 
 
__Problem__: Given an NxN grid, constuct two polygons with N integer coordinates. One should have the largest area and the other the smallest. The polygon must also follow these constraints.
<li>No two points may be in the same row or column</li>
<li>No intersection of lines</li>
<li>I probably forgot some. Check AZsPC for all rules.</li>

There were many Ns ranging from 5 to 512.
 
My answer to this problem was a brute force algorithm.
I originally was doing a genetic algorithm to solve the problem, but, since the constraints made the spectrum non-continuous, it made it hard for my genetic algorithms. They would always converge at a poor local minimum. \\\_(-.-)_/
 
In my final answer, I created an optimized brute force algorithm. It first finds the Delaunay triangles, then uses depth first graph search to determine the possible triangles, cutting off whenever the triangles becomes unfeasible. It is asynchronous and multi-cored. (In BruteForce.py)
 
This approach will find the smallest or largest triangles reliably on sizes under 15 in a couple of minutes. But it has a high order big O, so it will quickly bog down and has no real hope of completing the 512 on my measly laptop(or really any reasonable computer probably - I haven't done the math.)
 
If I were to optimize this algorithm further, I would need to find a way to reduce the order of the Big-O. One possible way would be to not try every possibile spread of the delaunay triangles, but rather make some guess as to whether a specific path would help or hurt the area of the triangle. For instance, in _general_, a very concave angle would more likely reduce area than a convex angle, and vice versa.
 
There is probably a better way to make a decision as to which path to follow, but I haven't thought of one. This problem was very similar to the travelling salesman problem: a certain number of nodes, path-finding, and optimization. There are great optimizations to that algorithm, so I'm sure there have to be many for this one too.