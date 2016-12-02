---
layout: page
description: "... without further ado, to see who is right."
---

Machine learning is a vast field advancing at an alarming rate.
I'll be using this site to highlight some of the resources 
I found useful in my struggle to keep up and share some 
work I've done to gain a deeper understanding of this fascinating topic. 

## Gibbs sampling for the uninitiated

![Graphical Model](https://ai2-s2-public.s3.amazonaws.com/figures/2016-11-01/fc024fbdc59c3b5e708268b29e00cebaf9593875/8-Figure4-1.png)

The first resource I'd like to highlight is the tutorial paper
[Gibbs Sampling for the Uninitiated](https://www.umiacs.umd.edu/~resnik/pubs/LAMP-TR-153.pdf)
by Resnik and Hardisty, which is a masterpiece of exposition.  Their main example provides an amazingly 
clear description of how to build a Gibbs sampler for the very simple Naı̈ve Bayes probabilistic model. 
I'll share an [implementation](https://nbviewer.jupyter.org/github/bobflagg/gibbs-sampling-for-the-uninitiated/blob/master/Gibbs-sampling-for-the-Uninitiated.ipynb) of Gibbs sampling in Python
and an [illutration](https://nbviewer.jupyter.org/github/bobflagg/gibbs-sampling-for-the-uninitiated/blob/master/a-gibbs-sampler-for-detecting-spam.ipynb) of how to detect spam with a the sampler.
Eventually, I'll demonstrate the power of Cython by optimizing the sampler to gain a couple of orders of magnitude in performance.

## "Let us calculate!"

![Let us calculate!](https://c1.staticflickr.com/6/5343/30599219720_a4e7fcd6ea_b.jpg)

The title of this site is taken from Leibniz's _The Art of Discovery_ (1685). This represents for me 
his vision that the only way to avoid controversy is to make our reasoning as clear as mathematics so that "... we can find 
our error at a glance, and when there are disputes among persons, we can simply say: Let us calculate [calculemus], without further ado, to see who is right."

