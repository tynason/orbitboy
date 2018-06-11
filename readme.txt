These use RGB coloring of 0 -1 which works in python 3 but not python 2.x, where it will look like nothing is plotting since it expects RGB values 0 - 255.
_________________________________________________

orbitboy.py

When illustrating fractals, everyone plots 2D color maps. Maps of the Mandelbrot set. Maps of the Julia set. And yes they are amazing. 

But each point on the map reduces the entire orbit of this point under the iterative recursion process to a scalar quantity, a colored pixel. The orbit contains a rich amount of detail that is dropped if you  map to a scalar. This script shows the orbits with statstics in greater detail. 

The script selects points matching a subjective "boring" criterion: After watching the evolution of various points (the orbits of those points), you see ones that are more interesting, or less boring.

Plenty of orbits are cool to "watch". Bound attractive orbits spiral into a central point. 
Bound repulsive ones repel endlessly around a central point.

What interested me were the orbits of points near the boundary of the set, which dally around for a long time before finally being torn to infinity under the tortures of the recursion process.

So the boring/non-boring pseudocode in the script is:

	def datagen(self,maxiterrs):
		contin=True
		boredcount=0 
		boring=True	# assume it's boring
		while contin:
			# pick a new p,q
			if boreme: # we don't care if it's boring so stop looking
				boring=False; contin=False
			else:
				# not boring if k large but not the max (long period escaper)
				if k>minbored and k<maxiterrs: # we found a non-boring one
					boring=False
			# if neither of these holds, boring is still True, we keep looking
			if not boring:
				contin=False

			return params,xstats,ystats,radstats,angstats,xdata,ydata,raddata,angdata,boredcount,ang2data

Discrete statistics are a little funny, you can never trust them like their analog relatives. If the sampling rate is too low you lose information and get aliasing, and if the time series is too short you will only ever get craggy looking data whether you are in direct or transform space. 

So by selecting a longer minbored you will get time series with a long sample and good sharp Fourier data. Which maybe you can trust. The minbored parameter means the orbit has to stick around at least that long to be interesting.

	maxiters=52000
	minbored=1200

What winds up being interesting in looking at these displays is the self-similarity in the statistics. The radius plot, its transform, looks similar to the angle plot or its transform.

The peaks in the Fourier spectrum look self-similar; but this is just what you would expect. Of course it must be so, since the dynamics of the orbit itself is self-similar.
_________________________________________________

orbitani.py

Similar but the radius and angle are plotted dynamically along with the orbit.
_________________________________________________

spiroboy.py

Plots spirograph orbits; the criterion I used to beauty here was sharp angles. To include "rounder" spirographs, just increase the borangle parameter.

	boring=False
	borangle=25