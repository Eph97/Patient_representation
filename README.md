# Simulate

* N = 100,000 people

* Real world is one draw of gammas from the prior distribution for each person

* How much we should update our guess of that person's treatment effects towards the overall ATE depends on what happens in a bunch of simulated worlds

* So now we make 1000 more columns, each of which are a simulated world, where we redraw gamma from the prior distribution for every individual

* In each simulated world, for each individual, we can compute beta_i -- their treatment effect -- and beta_ATE, the average treatment effect

* How much we update in the real-world depends on how beta_i and beta_ATE correlate across the 1000 simulations for each individual

* https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4259486


6206972597
