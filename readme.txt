orbitboy.py

When illustrating fractals, everyone plots 2D color maps. Maps of the Mandelbrot set. Maps of the Julia set. And yes they are amazing. 

But each point on the map reduces the entire orbit of this point under the iterative recursion process to a scalar quantity, a colored pixel. The orbit contains a rich amount of detail that is dropped if you map to a scalar. This script shows the orbits with statstics in greater detail. 

The script selects points matching a subjective "not boring" criterion: After watching the evolution of various points (the orbits of those points), you see ones that are more interesting, or less boring.

Plenty of orbits are cool to "watch". Bound attractive orbits spiral into a central point. 
Bound repulsive ones repel endlessly around a central point.

What interested me were the orbits of points near the boundary of the set, which dally around for a long time before finally being torn to infinity under the tortures of the recursion process.

So the boring/non-boring pseudocode in the script is:

	pick a new p,q
	boring=True
	iterate thru the orbit
		if it stays bound, it's boring
		if it escapes quickly, it's boring
		if it escapes only after a long time, it's interesting:
			plot the orbit
			plot histograms of rad (distance from origin) and external angle
			plot fourier transform of same to see periodicity

Now, discrete statistics are a little funny, you can never really trust them like their analog counterparts. If the sampling rate is too low you lose information and get aliasing, and if the time series is too short you will only ever get craggy looking data whether you are in direct or transform space. 

So by selecting a longer minbored you will get time series with a long sample and good sharp Fourier data. Which maybe you can trust. The minbored parameter means the orbit has to stick around at least that long to be interesting.

	maxiters=52000
	minbored=1200

What winds up being interesting in looking at these displays is the self-similarity in the statistics. The radius plot, its transform, looks similar to the angle plot or its transform.

The peaks in the Fourier spectrum look self-similar; but this is just what you would expect. Of course it must be so, since the dynamics of the orbit itself is self-similar.
_______________________________________

Notes on params:

	doani=True 	# animate the orbit, rad, and ang plots

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
	use mandel;
	
create table orbit (
orbit_id int not null auto_increment,
p double, q double, kappa int, maxrad float, radavg double,
raddev double, angavg double, angdev double, 
primary key (orbit_id));

Then put your pw in the code in place of YOURMYSQLPASSWORD.

A select will look like this:

	mysql> select * from orbit;
	+----------+------------------+-------------------+-------+--------+-------------------------+--------------------+----------------+----------------+----------------------+--------------------+---------------+---------------+
	| orbit_id | p                | q                 | kappa | maxrad | radmin                  | radmax             | radavg         | raddev         | angmin               | angmax             | angavg        | angdev        |
	+----------+------------------+-------------------+-------+--------+-------------------------+--------------------+----------------+----------------+----------------------+--------------------+---------------+---------------+
	|        1 | -0.695376385925  | -0.264296743298   | 16921 | 2.0    | 0.00006861256320827657  | 1.1603603593390344 | 0.314153715522 | 0.167233665709 | 3.852450431825563    | 344.5791295254192  | 195.573879397 | 24.0450394236 |
	|        2 | 0.357283220589   | 0.110374539755    |  1811 | 2.0    | 0.000049331173135135114 | 1.3262876438928033 | 0.348669849105 | 0.200517942487 | 2.02879523345997     | 356.79025242744865 | 49.101071176  | 39.005596492  |
	|        3 | 0.372323927482   | 0.17353751953     |  2333 | 2.0    | 0.00009705256305527462  | 1.1356668881659444 | 0.355739948964 | 0.204788939755 | 0.48455194749938857  | 359.87827481370795 | 59.190461302  | 44.5358542122 |
	|        4 | 0.28638115788    | -0.0153093978096  |  1527 | 2.0    | 0.0017103895890859862   | 1.4281968785914012 | 0.335656274152 | 0.190977611005 | 0.37958168900898653  | 359.99867140764616 | 321.047086202 | 73.0111101878 |
	+----------+------------------+-------------------+-------+--------+-------------------------+--------------------+----------------+----------------+----------------------+--------------------+---------------+---------------+
	4 rows in set (0.00 sec)
