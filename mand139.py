# mand00.py
# tnason 2018
#___________________________________________________________________#
import matplotlib.pyplot as plt
import pylab
import scipy.fftpack
import numpy as np
import random
import datetime
import time
import os
#___________________________________________________________________#

class Mandel(object):
	def __init__(self,boreme,minbored,xpos,ypos,wid,ht,figsleep,finalsleep):
		pass

	def getcolor(self):
		r=np.random.uniform(0,1)
		b=np.random.uniform(0,1)
		g=np.random.uniform(0,1)
		return (r,b,g)

	def lumengen(self,fore,back,grads): # get colors between fore and back color
		rangeRGB=[x1 - x2 for (x1,x2) in zip(fore,back)] # the range of RGB[0-1] to be covered
		segRGB=list(map(lambda x: x/grads,rangeRGB)) # the amount to decrement each element RGB
		R=np.zeros((grads,));G=np.zeros((grads,));B=np.zeros((grads,)) # start w/fore and decrement to back
		for nn in range(numgrads): R[nn]=fore[0]-nn*segRGB[0];G[nn]=fore[1]-nn*segRGB[1];B[nn]=fore[2]-nn*segRGB[2]
		return list(zip(R,G,B))

	def datagen(self,maxiterrs):
		contin=True
		boredcount=0 
		boring=True	# assume it's boring
		while contin:
			boredcount+=1
			params=[];xdata=[];ydata=[];raddata=[];angdata=[]
			xstats=[];ystats=[];radstats=[];angstats=[];ang2data=[]

			flip=x=np.random.random_integers(1,5)
			if flip==1: # 5 period attractive
				cx=np.random.uniform(0.355,0.360);cy=np.random.uniform(0.315,0.400)
			elif flip==2: # 5 period repulsive
				cx=np.random.uniform(-0.45,-0.6);cy=np.random.uniform(0.5,0.54)
			elif flip==3: # elephant valley
				cx=np.random.uniform(0.2,0.4);cy=np.random.uniform(0,0.4)
			elif flip==4: # seahorse valley
				cx=np.random.uniform(-0.8,-0.7);cy=np.random.uniform(0,0.1)
			elif flip==5: # MAIN CARDIOID
				theta=np.random.uniform(0,2*math.pi);rrr=np.random.uniform(0,0.05)
				cx=(0.5+rrr)*cos(theta)-cos(2*theta)/4;cy=(0.5+rrr)*sin(theta)-sin(2*theta)/4
			#elif flip==6: # BULB AT (-1.0)
			#	theta=np.random.uniform(0,2*math.pi);rrr=np.random.uniform(0,0.1)
			#	cx=(0.25+rrr)*cos(theta)-1;cy=(0.25+rrr)*sin(theta)
			#___________________________________________________________________#

			xdata.append(cx);ydata.append(cy)
			k=0;c=complex(cx,cy);z=complex(0.0,0.0)
			while abs(z)<2.0 and k<maxiterrs:
				k+=1;z=z*z+c
				xdata.append(z.real);ydata.append(z.imag)
			
			if boreme: # we don't care if it's boring so stop looking
				boring=False; contin=False
			else:
				# not boring if k large but not the max (long period escaper)
				if k>minbored and k<maxiterrs: # we found a non-boring one
					boring=False
			# if neither of these holds, boring is still True, we keep looking

			if not boring:
				contin=False
			
				for n in range(0,k-2):
					rad=xdata[n]**2+ydata[n]**2
					raddata.append(rad)
					if abs(xdata[n])>1e-8:
						ang=math.atan2(ydata[n],xdata[n]);ang=math.degrees(ang)%360
					else:
						ang=90
						print('arctan!!!!')
					angdata.append(ang)
				
					if doang:
						for n in range(0,k-2):  # internal angles
							point1=xdata[n],ydata[n];point2=xdata[n+1],ydata[n+1];point3=xdata[n+2],ydata[n+2]
							lineA=([point1[0],point1[1]]),([point2[0],point2[1]]);lineB=point2,point3
							vA=([(lineA[0][0]-lineA[1][0]),(lineA[0][1]-lineA[1][1])])
							vB=([(lineB[0][0]-lineB[1][0]),(lineB[0][1]-lineB[1][1])])
							dot_prod=dot(vA,vB);magA=dot(vA,vA)**0.5;magB=dot(vB,vB)**0.5

							argg=dot_prod/magB/magA
							if abs(argg)>1:
								print('arccos!!!!',argg)
								ang_deg=180
							else:
								ang=math.acos(argg)
								ang_deg=180-math.degrees(ang)%360
							ang2data.append(ang_deg)
						ang2data=ang2data[1:]

				params=[k,cx,cy]
				xmin=min(xdata[:k-trimend]);xmax=max(xdata[:k-trimend]);xavg=mean(xdata);xdev=std(xdata);xstats=[xmin,xmax,xavg,xdev]
				ymin=min(ydata[:k-trimend]);ymax=max(ydata[:k-trimend]);yavg=mean(ydata);ydev=std(ydata);ystats=[ymin,ymax,yavg,ydev]
				radmin=min(raddata);radmax=max(raddata);radavg=mean(raddata);raddev=std(raddata);radstats=[radmin,radmax,radavg,raddev]
				angmin=min(angdata);angmax=max(angdata);angavg=mean(angdata);angdev=std(angdata);angstats=[angmin,angmax,angavg,angdev]
				return params,xstats,ystats,radstats,angstats,xdata,ydata,raddata,angdata,boredcount,ang2data

			else: print('BORING #',boredcount,'kappa=',k,'\tcx=',cx,'\tcy=',cy)

	def plotme(self):

		def refreshall(self):
			for ax in fig.axes:
				currcolor = ax.patch.get_facecolor();currtitle = ax.get_title()
				ax.clear();ax.patch.set_facecolor(currcolor)
				ax.set_title(currtitle,fontsize=5,color=mybritegrn)
				ax.grid(True);ax.patch.set_alpha(1.0)
				ax.tick_params(axis='x',labelsize=6,labelcolor='#ffffff')
				ax.tick_params(axis='y',labelsize=6,labelcolor='#ffffff')

		def refreshone(self,n):
			ax=fig.axes[n]
			currcolor = ax.patch.get_facecolor();currtitle = ax.get_title()
			ax.clear();ax.patch.set_facecolor(currcolor)
			ax.set_title(currtitle,fontsize=5,color=mybritegrn)
			ax.grid(True);ax.patch.set_alpha(1.0)
			ax.tick_params(axis='x',labelsize=6,labelcolor='#ffffff')
			ax.tick_params(axis='y',labelsize=6,labelcolor='#ffffff')

		fig = plt.figure()
		ax0 = plt.subplot2grid((4,6),(0,0),colspan=2,rowspan=2)

		ax1 = plt.subplot2grid((4,6),(2,0))
		ax2 = plt.subplot2grid((4,6),(2,1))
		ax3 = plt.subplot2grid((4,6),(3,0))
		ax4 = plt.subplot2grid((4,6),(3,1))

		ax5 = plt.subplot2grid((4,6),(0,2),colspan=2)
		ax6 = plt.subplot2grid((4,6),(0,4),colspan=2)
		ax7 = plt.subplot2grid((4,6),(1,2),colspan=2)
		ax8 = plt.subplot2grid((4,6),(1,4),colspan=2)

		ax9 = plt.subplot2grid((4,6),(2,2),colspan=2)
		ax10 = plt.subplot2grid((4,6),(2,4),colspan=2)
		ax11 = plt.subplot2grid((4,6),(3,2),colspan=2)
		ax12 = plt.subplot2grid((4,6),(3,4),colspan=2)

		# initial subplot settings
		for ax in fig.axes:
			ax.grid(True)
			ax.patch.set_facecolor(rgbback)
			ax.patch.set_alpha(1.0)
			ax.tick_params(axis='x',labelsize=6,labelcolor='#ffffff')
			ax.tick_params(axis='y',labelsize=6,labelcolor='#ffffff')		

		win = plt.gcf().canvas.manager.window
		fig.canvas.manager.window.wm_geometry('%dx%d%+d%+d' % (wid,ht,xpos,ypos))
		fig.patch.set_facecolor(rgbback)
		plt.tight_layout(pad=0.1,w_pad=0.1,h_pad=0.5)
		fig.subplots_adjust(top=0.935)

		contin=True
		nice=[];xnice=[];ynice=[] #list of non-boring orbits

		while contin==True:
			start_time=time.time()
			result=self.datagen(maxiters)
			rgbfore=self.getcolor()

			lumon=(255,200,0)
			lumon=list(map(lambda x: x/256,lumon))
			lumoff=(0,125,125)
			lumoff=list(map(lambda x: x/256,lumoff))
			lumen=self.lumengen(lumon,lumoff,numgrads)
			alumen=lumen[::-1]

			params=result[0]
			kappa=params[0];p=params[1];q=params[2];boredcount=result[9]

			if kappa>numgrads:
				lag=int(kappa/numgrads)
			else:
				lag=1
			print('kappa,numgrads,lag',kappa,numgrads,lag)

			ang2data=result[10]
			xdata=result[5];ydata=result[6];raddata=result[7];angdata=result[8]
			xmin=result[1][0];xmax=result[1][1];ymin=result[2][0];ymax=result[2][1]
			mytitle='Mandelbrot p= '+str(p)+' q= '+str(q)+' kappa= '+str(kappa)
			fig.suptitle(mytitle,fontsize=9,color='#00ff00')
			print('\nSELECTED non-boring p,q after',str(boredcount),'tries:')
			print('\tcx='+str(p)+';cy='+str(q))
			print('EXITED AT:')
			print('\tkappa:',kappa);print('\tminbored:',minbored);print('\tmaxiters:',maxiters)
			print('\tboreme:',boreme);print('\tnumbins:',numbins);print('\ttrimend:',trimend);print('\tlag:',lag)

			refreshall(self)
			
			end1=int(kappa/10);end2=int(kappa/5);end3=int(kappa/2);end4=int(0.9*kappa)
			lumend1=alumen[int(numgrads/10)]
			lumend2=alumen[int(numgrads/5)]
			lumend3=alumen[int(numgrads/2)]
			lumend4=alumen[int(numgrads)-1]
			
			ax1.set_title('first 1/10',loc='left',fontsize=8,color=mybritegrn)
			ax1.plot(xdata[0:end1],ydata[0:end1],lw=linewid,color=lumend1)
				
			ax2.set_title('1/10 - 1/5',loc='left',fontsize=8,color=mybritegrn)
			ax2.plot(xdata[end1:end2],ydata[end1:end2],lw=linewid,color=lumend2)
			
			ax3.set_title('1/5 - 1/2',loc='left',fontsize=8,color=mybritegrn)
			ax3.plot(xdata[end2:end3],ydata[end2:end3],lw=linewid,color=lumend3)
			
			ax4.set_xlim(xmin-0.8*abs(xmin),xmax+0.8*abs(xmax));ax4.set_ylim(ymin-0.8*abs(ymin),ymax+0.8*abs(ymax))
			ax4.set_title('last 1/10  trimend=0',loc='left',fontsize=8,color=mybritegrn)
			ax4.plot(xdata[end3:],ydata[end3:],lw=linewid,color=lumend4)

			ax5.set_title('rad-k',loc='left',fontsize=8,color=mybritegrn)
			ax5.plot(raddata,lw=linewid,color=mybritegrn)

			ax7.set_title('ang-k',loc='left',fontsize=8,color=mybritegrn)
			ax7.plot(angdata,lw=linewid,color=mygreen)

			ax9.set_title('rad-k last 1/10',loc='left',fontsize=8,color=mybritegrn)
			ax9.plot(raddata[end4:],lw=linewid,color=myturq)

			ax10.set_title('rad hist',loc='left',fontsize=8,color=mybritegrn)
			ax10.set_xlabel('bin (rad value)',fontsize=8,color=mybritegrn)
			ax10.set_ylabel('rad hist # in bin',fontsize=8,color=mybritegrn)			
			ax10.hist(raddata,bins=numbins,normed=True,color=myyell)

			ax11.set_title('ang-k last 1/10',loc='left',fontsize=8,color=mybritegrn)
			ax11.plot(angdata[end4:],lw=linewid,color=myteal)
			
			ax12.set_title('ang hist',loc='left',fontsize=8,color=mybritegrn)
			ax12.set_xlabel('bin (ang value)',fontsize=8,color=mybritegrn)
			ax12.set_ylabel('ang hist # in bin',fontsize=8,color=mybritegrn)			
			ax12.hist(angdata,bins=numbins,normed=True,color=myyell2)

			#___________________________________________________________________#
			data=raddata
			datalbl='rad'
			maxfreq=int(kappa/15)+1

			Fs = 300  # sampling rate - HIGHER = SHARPER PEAK
			Ts = 1.0/Fs # sampling interval
			n=kappa-2 # length of the signal
			kk = np.arange(kappa-2)
			T = kappa/Fs
			frq2 = kk/T # two sides frequency range
			frq1 = frq2[1:int(n/2)] # one side frequency slice
			Y2 = 2*np.fft.fft(data)/n # fft computing and norm
			Y1 = Y2[1:int(n/2)]


			#ax1.set_title('xmap-ymap',fontsize=8,color=mybritegrn)
			#ax1.scatter(xnice,ynice,s=1,lw=linewid,color=myorange)

			# if doang...
			#ax1111.set_title('ang2int-k',fontsize=8,color=mybritegrn)
			#ax1111.plot(ang2data,lw=linewid,color=myturq)
			

			#ax11.set_title('Fsamp=300hz t=1sec',fontsize=8,color=mybritegrn)
			#ax11.set_xlabel('k (iters/time)',fontsize=8,color=mybritegrn)
			#ax11.set_ylabel(datalbl+' (amplitude)',fontsize=8,color=mybritegrn)
			#ax11.plot(kk,data,lw=linewid,color=mybritegrn)

			#ax12.set_ylabel(datalbl+' histogram',fontsize=8,color=mybritegrn)
			#ax12.hist(data,bins=numbins,normed=True,color=myorange2)

			ax6.set_title('freq vs FT('+datalbl+')  Fsamp=300',loc='left',fontsize=8,color=mybritegrn)
			ax6.set_xlabel('freq (Hz)',fontsize=8,color=mybritegrn)
			ax6.set_ylabel('FT('+datalbl+')',fontsize=8,color=mybritegrn)
			ax6.plot(frq1[1:maxfreq],abs(Y1[1:maxfreq]),lw=linewid,color=myorange)

			data=angdata
			datalbl='ang'
			maxfreq=int(kappa/15)+1

			Fs = 300  # sampling rate - HIGHER = SHARPER PEAK
			Ts = 1.0/Fs # sampling interval
			n=kappa-2 # length of the signal
			kk = np.arange(kappa-2)
			T = kappa/Fs
			frq2 = kk/T # two sides frequency range
			frq1 = frq2[1:int(n/2)] # one side frequency slice
			Y2 = 2*np.fft.fft(data)/n # fft computing and norm
			Y1 = Y2[1:int(n/2)]

			ax8.set_title('freq vs FT('+datalbl+')  Fsamp=300',loc='left',fontsize=8,color=mybritegrn)
			ax8.set_xlabel('freq (Hz)',fontsize=8,color=mybritegrn)
			ax8.set_ylabel('FT('+datalbl+')',fontsize=8,color=mybritegrn)
			ax8.plot(frq1[1:maxfreq],abs(Y1[1:maxfreq]),lw=linewid,color=myorange2)			
			#___________________________________________________________________#
			
			plt.show(block=False);fig.canvas.draw()

			ax0.set_xlim(xmin,xmax);ax0.set_ylim(ymin,ymax)
			ax0.set_title('x-y orbit plot  trimend='+str(trimend),loc='left',fontsize=8,color=mybritegrn)
			if dodata:
				ax0.plot(xdata[0:kappa-trimend],ydata[0:kappa-trimend],lw=linewid,color=myteal)

			else:
				if doforward:

					# The -1 in xdata[n*lag-1:(n+1)*lag] is needed to make the line from the slice start from the last point in the previous slice
					# so the orbit is connected which can be seen by print(n,n*lag-1,(n+1)*lag,xdata[n*lag-1:(n+1)*lag])
					# But for the first slice this results in the slice [-1:0] which means it tries to start at the last element "thru" the first
					# so the first slice is plotted manually above.
					#
					# At the last slice, lag=int(kappa/numgrads) may not divide evenly into kappa
					# so there will typically be a few points at the end not covered by the loop;
					# these are handled manually after the loop.

					#print(xdata)
					ax0.set_xlim(xmin,xmax);ax0.set_ylim(ymin,ymax)

					n=0
					#print('\nFIRST SLICE') # handle first slice manually
					#print('n,n*lag,(n+1)*lag,xdata[n*lag:(n+1)*lag]')
					#print(n,n*lag,(n+1)*lag,xdata[n*lag:(n+1)*lag])					
					ax0.plot(xdata[n*lag:(n+1)*lag],ydata[n*lag:(n+1)*lag],lw=linewid,color=alumen[n])
					plt.show(block=False);fig.canvas.draw()
					time.sleep(linesleep)

					#for n in range(1,numgrads):
					while (n+1)*lag<kappa-lag:
						n+=1
						#refreshone(self,0)
						#plt.show(block=False);fig.canvas.draw()

						#print('n,n*lag-1,(n+1)*lag,xdata[n*lag-1:(n+1)*lag]')
						#print(n,n*lag-1,(n+1)*lag,xdata[n*lag-1:(n+1)*lag])

						ax0.plot(xdata[n*lag-1:(n+1)*lag],ydata[n*lag-1:(n+1)*lag],lw=linewid,color=alumen[n])
						plt.show(block=False);fig.canvas.draw()
						time.sleep(linesleep)

					n=numgrads-1
					#print('LAST ITERATES') # handle last iterates manually
					#print('n,(n+1)*lag-1,len(xdata),xdata[(n+1)*lag-1:]'')
					#print(n,(n+1)*lag-1,len(xdata),xdata[(n+1)*lag-1:])
					ax0.plot(xdata[(n+1)*lag-1:],ydata[(n+1)*lag-1:],lw=linewid,color=alumen[n])
					plt.show(block=False);fig.canvas.draw()
					time.sleep(figsleep)

			nice.append(params);xnice.append(params[1]);ynice.append(params[2])
			#print('NICE!!!\n',nice)
			intfile='nice'+datetime.datetime.now().strftime("%Y%m%d_%H%M")+'.txt'
			if dointerest: # SAVE NICE ORBITS
				if os.path.isdir('imgs'):intfile='imgs/'+intfile
				#intdata='kappa='+str(kappa)+'\ncx='+str(p)+';cy='+str(q)+'\n'
				intdata=''
				for el in nice:
					csvrow=str(el[0])+','+str(el[1])+','+str(el[2])+'\n'
					intdata+=str(csvrow)
				#print('INTDATA!!!\n',intdata)
				with open(intfile,'w') as intfile:intfile.write(intdata)
			
			if dosave:  # SAVE PARAMS n PNG
				outfile=str(kappa)+'_'+str(round(p,digits))+'_'+str(round(q,digits)) #print(outfile)
				if kappa>4000:outfile='BIGKAPPA'+outfile
				if os.path.isdir('imgs'):outfile='imgs/'+outfile
				outpic=outfile+'.png'; savefig(outpic,facecolor=fig.get_facecolor(),bbox_inches='tight')
				outdata='kappa='+str(kappa)+'\ncx='+str(p)+'\ncy='+str(q)
				outfile=outfile+'.txt'
				with open(outfile,'w') as outfile:	outfile.write(outdata)

			time.time();elapsed=time.time()-start_time
			timerpt='\telapsed: '+str(elapsed);print(timerpt)
			time.sleep(figsleep)
			if not doloop:contin=False

		if figclose:
			time.sleep(finalsleep)
			plt.close()
		else:
			plt.show(block=True)
#___________________________________________________________________#
figsleep=0
finalsleep=0
linesleep=0

wid=1000;ht=700;xpos=10;ypos=100
wid=1800;ht=1100;xpos=0;ypos=0
wid=1200;ht=650;xpos=2100;ypos=100

trimend=4
maxiters=52000

minbored=1200
numbins=200
numgrads=20
maxrad=10000

dodata=False
dosave=True
dointerest=True
doang=False
boreme=False;figclose=True
linewid=.6;figscale=1.0;digits=7
doloop=True;doforward=True;colorforward=True
doprint=True;doprintreveal=True;doprintappend=True;doprintall=False

mygunmet='#113344';mygunmet2='#052529';myblue='#11aacc';mydkblue='#0000cc'
myturq='#00ffff';myturq2='#11bbbb';myteal='#00ffcc';myteal2='#00ccaa'
mygreen='#44bb44';mybritegrn='#00ff66';myfuscia='#ff00ff';myfuscia2='#dd0099'
mypurp='#ff00cc';mypurp2='#9933ff';myred='#cc0000';myorange='#ffaa00'
myorange2='#ff6600';myyell='#ffff00';myyell2='#ffcc00';rgbback=mygunmet2

mymand=Mandel(boreme,minbored,xpos,ypos,wid,ht,figsleep,finalsleep)
mymand.plotme()
#___________________________________________________________________#

"""
# 5 period attractive
cx=np.random.uniform(0.355,0.360);cy=np.random.uniform(0.315,0.400)
	cx=0.35548881306973545;cy=0.3341487089982737 ### 5attrSWEET ###
	cx=0.377270309029;cy=-0.182567450517
	cx=0.35600423;cy=0.33093
	cx=0.36126078;cy=0.320503
# 5 period repulsive
cx=np.random.uniform(-0.45,-0.6);cy=np.random.uniform(0.5,0.54)
	cx=-0.5206501526991798;cy=0.5095722298589745 ### 5repSWEET ###
	cx=-0.527273;cy=0.52851
	cx=-0.4806341;cy=0.532485
	cx=-0.453318;cy=0.55599

# elephant valley
cx=np.random.uniform(0.2,0.4);cy=np.random.uniform(0,0.4)
	cx=0.37719262229219286;cy=0.18737276071403622 ### eleSWEET ###
		cx=np.random.uniform(0.37,0.38);cy=np.random.uniform(0.18,0.19)
	cx=0.260239040369;cy=0.0020916862844
# seahorse valley
cx=np.random.uniform(-0.8,-0.7);cy=np.random.uniform(0,0.1)
	cx=-0.747853645833;cy=0.062355588549 ### seaSWEET ###

# BULB AT (-1.0)
theta=np.random.uniform(0,2*math.pi);rrr=np.random.uniform(0,0.1)
cx=(0.25+rrr)*cos(theta)-1;cy=(0.25+rrr)*sin(theta)
	cx=-0.7484815173001504;cy=0.0532662111705495 ### bowtieSWEET  ###
	cx=-0.735421537445;cy=-0.163709777921 ### bowtieSWEET  ###

# MAIN CARDIOID
theta=np.random.uniform(0,2*math.pi);rrr=np.random.uniform(0,0.05)
cx=(0.5+rrr)*cos(theta)-cos(2*theta)/4;cy=(0.5+rrr)*sin(theta)-sin(2*theta)/4
	cx=0.250214148139;cy=-4.79572217515e-06 ### cardioSWEET ###

# rand around single point
cx=-0.5206501526991798;cy=0.5095722298589745
rrrx=np.random.uniform(-0.025,0.025);rrry=np.random.uniform(-0.025,0.025)
cx=cx+rrrx;cy=cy+rrry

# big rectangle
cx=np.random.uniform(-3,.8);cy=np.random.uniform(0,2)

# SPIRO
# can put in spiro params here

## sweet SILHOUTTE
cx=0.252646183022;cy=-0.000235409650954

### fourierSWEET  ###
cx=0.355384646380999;cy=0.3332026372971109
cx=0.3563268958199877;cy=0.33007286156906823
cx=0.3581128676046881;cy=0.3244716896271824
cx=-0.735421537445;cy=-0.163709777921 how in hell is ther only 1 peak at 26 ?
cx=-0.7506095013167247;cy=0.02017840558087991
cx=-0.7451660763608009;cy=0.08098427487766885
cx=-0.748788372883687;cy=0.0692479509378728
cx=-0.5013475449887075;cy=0.516886282748629
cx=-0.0170723831006;cy=0.639306902532
"""
