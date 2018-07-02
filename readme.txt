orbitboy.py

When illustrating fractals, everyone plots 2D color maps. Maps of the Mandelbrot set. Maps of the Julia set. And yes they are amazing. 

But each point on the map reduces the entire orbit of this point under the iterative recursion process to a scalar quantity, a colored pixel. The orbit contains a rich amount of detail that is dropped if you map to a scalar. This script shows the orbits with statstics in greater detail. 

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

_______________________________________

Notes on params:

	doani=True 	# animate the orbit, rad, ang and histogram plots

This will animate the orbit, radius and angle plots but does slow down the plotting. I'm not using any blt or the canned animation routines in python just a manual slicing so set to False if you want.


	mygunmet='#113344';mygunmet2='#052529'

The default colors are defined in hex but are applied as RGB coloring of 0 -1 which works in python 3 but not python 2.x, where it will look like nothing is plotting since it expects RGB values 0 - 255. So please use python 3.x.


	maxiters=52000	# max iterations
	minbored=1000	# minimum non-boring orbit iterations
	maxrad=2.0		# defines the escape criterion
	trimend=4		# omits the final few iterations from some of the plots
	numbins=200		# no. of bins in the histograms

	numgrads=20 	# how many slices to plot the animated orbit
	boreme=False 	# pick a long orbit which escapes at the end

These are common mandelbrot params except for the boring criteria.


	doang=False

This is for internal angles, not implemented yet.


	dosave=True # save params to DB or file, and save png

If you set dosave=True the script will save a png to a subfolder imgs or in the folder you are running from.
It will also save the params to a mysql db if it exists, else it will save a text file with the params along with the png.

You can create the mysql db as follows:

	create database mandel;

	create table orbit (
	orbit_id int not null auto_increment,
	p varchar(20),
	q varchar(20),
	kappa int(10),
	maxrad varchar(10),
	primary key (orbit_id)
	);

Then put your pw in the code in place of YOURMYSQLPW.

A select will look like this:

	mysql> select * from orbit;
	+----------+---------------------+----------------------+-------+--------+
	| orbit_id | p                   | q                    | kappa | maxrad |
	+----------+---------------------+----------------------+-------+--------+
	|        1 | 0.2662964889352877  | 0.004143476663879161 |  5765 | 2.0    |
	|        2 | -0.672220295245     | -0.319530729422      |   909 | 2.0    |
	|        3 | 0.3591546393348771  | 0.32320593617103527  |  1629 | 2.0    |
	|        4 | -0.7660998450315548 | 0.08857332208333829  | 15249 | 2.0    |
	+----------+---------------------+----------------------+-------+--------+
	4 rows in set (0.00 sec)
