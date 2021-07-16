# Zbrodoff-In-Nengo

This is a neural simulation of the [Zbrodoff experiment](https://link.springer.com/content/pdf/10.3758%2FBF03200922.pdf) implemented in [Nengo](https://www.nengo.ai/). The models are supposed to verify simple equations such as A+2=C. You can read the [presentation](./Presentation.pdf) if you're interested in learning more, or the [thesis report](./report/report.pdf) if you're _really_ interested.



## Requirements

- [Python 3](https://www.python.org/)
- [Nengo](https://www.nengo.ai/) (`$ pip install nengo`)
- [Nengo SPA](https://www.nengo.ai/nengo-spa/) (`$ pip install nengo-spa`)
- [Matplotlib](https://matplotlib.org/) (`$ pip install matplotlib`)
- [NumPy](https://numpy.org/) (`$ pip install numpy`)
  - **Optional**: if you have a fast GPU and you're going to run the models many times, it might be worthwhile to install [nengo-ocl](https://pypi.org/project/nengo-ocl/). The setup for this is more involved, so only install it if you really need it.



You shouldn't need to worry about installing particular versions of the above, but in case you do here is what I used: Python 3.8.5, nengo 3.1.0, nengo-spa 1.2.0, matplotlib 3.3.3, numpy 1.19.5, nengo-ocl 2.1.0.  



## How to run

- `$ python3 combined_model.py` to run the combined counting + learning model.
- `$ python3 counting_model.py` to run the counting model.
- `$ python3 learning_model.py` to run the learning model.
- `$ python3 plot_timing.py` produces a response time comparison between this model, ACT-R, and human experimental data.



## How to reproduce figures from paper

Running the models without modification should produce the same figures from the paper. Whenever a model is finished with a problem, a timestamp will be printed out - you can use these to recreate the response times hardcoded in `plot_timing.py`.
