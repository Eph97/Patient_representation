# Next Steps on model


I think the next step is to try to generalize in a few directions (in this order!):

1. A first point to note is that the case you solved (the one I wanted) is NOT nested by beta_i = x*gamma with independent gammas.
	Specifically, you solved:

$$\beta_i = (1-x)\gamma_0 + x*\gamma_1 = \gamma_0 + x*(\gamma_1-\gamma_0)$$
So if we write this as $\beta_i = x*\gamma$, the constant and coefficient on x would be correlated through $\gamma_0$.

We deliberately did that to impose symmetry so it wasn’t the case that, e.g., men had higher variance of beta_i than women.

2. So, one alternative case to solve is beta_i = gamma_0 + x*gamma_1, with gamma_0 and gamma_1 independent. I think solve that case next and let’s see if we get a similar result. Write up the same way.
 
3. A generalization of 1 and 2 would be, beta_i = gamma_0 + x*gamma_1, allowing for arbitrary correlation between gamma_0 and gamma_1 (and thus nesting case 1). Solve this after 2). Write up the same way.

4. A generalization of 2) (but not 3) would be beta_i = x*gamma, where x includes k variables, not just a constant and a single x. This should be solved with matrix algebra. This is the case you initially tried to solve. There are two special cases to consider:

	a. We only allow independent drawing of types. That is, the representation in the trial is just a vector (or a diagonal matrix), where we say the trial will be 70% women and 30% black but independently drawn (so it’s 21% black women). This is I think what you started trying to solve earlier.
	b. We solve the more general case, where the solution is the proportion of each possible intersection of types. So we separately specify the proportion of black women. Now the solution looks like a matrix rather than a vector.

After 3), solve 4a) and 4b).

5. Whereas 4) is a generalization of 2), we also can generalize 3) in the same way, also allowing for correlations between the gammas. This may well be the theorem in the actual paper.

6. We could potentially relax the assumption that x is binary, allowing for continuous x’s. I’m not sure this is critical since a sufficient number of binary x’s can always approximate arbitrarily closely any continuous x. But it might be nice to include if it is tractable.

# Follow up

And one more future generalization while we’re at it is to finite samples in the experiment (but don’t worry about that until we’ve solved everything else)

1. Suppose women are the minority in the population. The fewer women you include, the more you update (and get accurate beta) for men relative to women. This force says just have men.

2. Because the objective function is convex, bigger errors matter more. So for example, if you are far off for women and close for men, you’d rather make beta more accurate for women. This force says don’t just have men.
 
For some reason, 1) always seems to dominate over 2). I would have expected an interior solution. That is, I would have thought that if the population is 30% women and 70% men, that you can do better than having zero women in the trial because at that point, you’d get things exactly right for men but have huge error for women, so there would be first-order benefits for women and only second order losses for men of adding more women.

If your solution is right, that intuition is wrong, but we should examine the math to try to understand why.

# Follow up 2
A quick follow-up to this. I think it would be helpful to develop intuition to graph the error and squared error for men and women separately as a function of xbar in a case where the minority group is something like 25% of the population. This graph will help give a sense of why the gain to the minority group never outweighs the loss to the majority group. It would be helpful to see the two error functions on the same graph with the same scale.


# Notes
* MSE = E_gamma [ p(beta_i,men-beta_i,men,post)^2 +(1-p)(beta_i,women-beta_i,women,post)^2]
* W_men = beta_i,men-beta_i,men,post doesn't depend on x
* There is no such thing as: E_x(W^2_men) 
* E_x(W^2_men) = W^2_men
* MSE = E_gamma E_x (beta_i-beta_post)^2
* E_x (beta_i-beta_post)^2 = p(beta_i,men-beta_post,men)^2 + (1-p)(beta_i,women-beta_i,post,women)^2
* W_men = beta_i,men-beta_post,men
* MSE = E_gamma [ p(W_men)^2 +(1-p)(W_women)^2 ] 
* Conjecture: W_men+W_women = 0 
* (or maybe 1) 
* MSE = E_gamma [ p(W_men)^2 +(1-p)(1-W_men)^2 ] 
