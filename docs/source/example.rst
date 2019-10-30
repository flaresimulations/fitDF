Example usage
=============

In this tutorial we'll demonstrate using fitDF on a galaxy luminosity function. First, we can generate some fake observational data (counts per luminosity bin) by running

.. highlight::bash
    python generate_fake_observations.py

This produces two files, :code:`fake_observations.json` and :code:`input_parameters.json`, which contain the counts and the chosen input parameters, respectively.

Now we can run our fit:


.. highlight::bash
    python example_fit.py

this performs MCMC to fit the observations using a single Schechter function model (identical to that used in the data generation). Once the sampling is complete, two plots are produced. :code:`triangle.png` shows a triangle (corner) plot of the posterior parameter distributions, with the original parameters shown as the horizontal lines on each marginal distribution.

.. image::triangle.png

:code:`LF.png` shows the fitted luminosity function. The original and fitted functions are both shown, as well as the original data.

.. image::LF.png



You can experiment with the contents of :code:`example_fit.py`. For example, try changing the number of samples or burn-in period, or even adjust the prior parameter distributions. :code:`example_fit.py` can be used as a template for your own projects.
