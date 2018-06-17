# orbitboy.py
# mand157.py
# tnason 2018
# https://github.com/tynason/orbitboy
#___________________________________________________________________#
import matplotlib.pyplot as plt
from pylab import *
import scipy.fftpack
import numpy as np
import random
import datetime
import time
import os
import mysql.connector
from mysql.connector import errorcode
#___________________________________________________________________#

class Mandel(object):
	def __init__(self,boreme,minbored,xpos,ypos,wid,ht,figsleep,finalsleep):
		pass

	def autocorr(self,xx):
		result = np.correlate(xx, xx, mode='full')
		return result[result.size/2:]

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

			#flip=x=np.random.random_integers(1,5)
			flip=5

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
			while abs(z)<maxrad and k<maxiterrs:
				k+=1;z=z*z+c
				xdata.append(z.real);ydata.append(z.imag)
			
			self.mandfunc='z*z+c'

			"""
			# other functions
			while abs(z)<2.0 and k<maxiterrs:
				k+=1;z=z**2*(1-z)/(1+z)
				xdata.append(z.real);ydata.append(z.imag)
			self.mandfunc='z**2*z.real*(1+z)/(1-z)'

			while k<maxiterrs and x**2+y**2<maxrad:
				k+=1;
				denom=x**2+y**2+.01
				x=((1-x)*(x**2-y**2)+8*x*y)/denom+cx
				y=(2*x*(1-x*y)-(x**2+y**2))*(y/denom)+cy
				xdata.append(x);ydata.append(y)
			self.mandfunc='moth4'			
			
			while k<maxiterrs and x**2+y**2<maxrad:
				k+=1;
				denom=x**2+y**3+.2
				x=((1-x)*(x**2-y**2)+2*x*y**2)/denom+cx
				y=(2*x*(1-x)-(x**2+y**2))*(y/denom)+cy
				xdata.append(x);ydata.append(y)
			self.mandfunc='moth3'
			"""

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

					# external angle
					if abs(xdata[n])>1e-8:
						ang=math.atan2(ydata[n],xdata[n]);ang=math.degrees(ang)%360
					else:
						ang=90
						print('arctan!!!!')
					angdata.append(ang)
				
					if doang:
						for n in range(0,k-2):  # internal angle; needs TRY EXCEPT
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
				currcolor = ax.patch.get_facecolor();currtitle = ax.get_title(loc='left')
				currxlabel=ax.get_xlabel();currylabel=ax.get_ylabel()
				currxlim=ax.get_xlim();currylim=ax.get_ylim()
				ax.clear();ax.patch.set_facecolor(currcolor)
				ax.set_title(currtitle,loc='left',fontsize=8,color=mybritegrn)
				ax.grid(True);ax.patch.set_alpha(1.0)
				ax.tick_params(axis='x',labelsize=6,labelcolor='#ffffff')
				ax.tick_params(axis='y',labelsize=6,labelcolor='#ffffff')
				ax.set_xlabel(currxlabel,fontsize=8,color=mybritegrn)
				ax.set_ylabel(currylabel,fontsize=8,color=mybritegrn)
				#ax.set_xlim(currxlim);ax.set_ylim(currylim)

		def refreshone(self,n):
			ax=fig.axes[n]
			currcolor = ax.patch.get_facecolor();currtitle = ax.get_title(loc='left')
			currxlabel=ax.get_xlabel();currylabel=ax.get_ylabel()
			currxlim=ax.get_xlim();currylim=ax.get_ylim()
			ax.clear();ax.patch.set_facecolor(currcolor)
			ax.set_title(currtitle,loc='left',fontsize=8,color=mybritegrn)
			ax.grid(True);ax.patch.set_alpha(1.0)
			ax.tick_params(axis='x',labelsize=6,labelcolor='#ffffff')
			ax.tick_params(axis='y',labelsize=6,labelcolor='#ffffff')
			ax.set_xlabel(currxlabel,fontsize=8,color=mybritegrn)
			ax.set_ylabel(currylabel,fontsize=8,color=mybritegrn)
			ax.set_xlim(currxlim);ax.set_ylim(currylim)

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

			# if numgrads>kappa, you are trying to plot more slices than there are data points so set lag=1
			if kappa>numgrads:
				lag=int(kappa/numgrads)
			else:
				lag=1

			xdata=result[5];ydata=result[6];raddata=result[7];angdata=result[8]
			xmin=result[1][0];xmax=result[1][1];ymin=result[2][0];ymax=result[2][1]
			radmin=result[3][0];radmax=result[3][1]
			angmin=result[4][0];angmax=result[4][1]
			ang2data=result[10]

			mytitle='Mandelbrot ' +self.mandfunc+ '  p= '+str(p)+'  q= '+str(q)+'  kappa= '+str(kappa)+'  maxrad= '+str(maxrad)
			fig.suptitle(mytitle,fontsize=9,color='#00ff00')

			print('\nSELECTED non-boring p,q after',str(boredcount),'tries:')
			print('\tcx='+str(p)+';cy='+str(q))
			print('EXITED AT:')
			print('\tkappa:',kappa);print('\tminbored:',minbored);print('\tmaxiters:',maxiters)
			print('\tboreme:',boreme);print('\tnumbins:',numbins);print('\ttrimend:',trimend);print('\tlag:',lag)
			print('\tnumgrads',numgrads)

			refreshall(self)
			
			end1=int(kappa/10);end2=int(kappa/5);end3=int(kappa/2);end4=int(0.9*kappa)
			lumend1=alumen[int(numgrads/10)]
			lumend2=alumen[int(numgrads/5)]
			lumend3=alumen[int(numgrads/2)]
			lumend4=alumen[int(numgrads)-1]

			ax0.set_title('x-y orbit plot  trimend='+str(trimend),loc='left',fontsize=8,color=mybritegrn)

			ax1.set_title('first 1/10',loc='left',fontsize=8,color=mybritegrn)
			ax1.plot(xdata[0:end1],ydata[0:end1],lw=linewid,color=lumend1)
				
			ax2.set_title('1/10 - 1/5',loc='left',fontsize=8,color=mybritegrn)
			ax2.plot(xdata[end1:end2],ydata[end1:end2],lw=linewid,color=lumend2)
			
			ax3.set_title('1/5 - 1/2',loc='left',fontsize=8,color=mybritegrn)
			ax3.plot(xdata[end2:end3],ydata[end2:end3],lw=linewid,color=lumend3)

			ax4.set_xlim(xmin-0.8*abs(xmin),xmax+0.8*abs(xmax))
			ax4.set_ylim(ymin-0.8*abs(ymin),ymax+0.8*abs(ymax))
			ax4.set_title('last 1/10  trimend=0',loc='left',fontsize=8,color=mybritegrn)
			ax4.plot(xdata[end3:],ydata[end3:],lw=linewid,color=lumend4)
			
			ax5.set_title('rad vs k',loc='left',fontsize=8,color=mybritegrn)
			ax7.set_title('ang vs k',loc='left',fontsize=8,color=mybritegrn)

			#ax9.set_title('rad vs k last 1/10',loc='left',fontsize=8,color=mybritegrn)
			#ax9.plot(raddata[end4:],lw=linewid,color=myturq)

			# ax9  autocorrelation
			auto=self.autocorr(raddata)
			ax9.set_title('autocorrelation(rad)',loc='left',fontsize=8,color=mybritegrn)
			ax9.set_xlim(0,150)
			ax9.plot(auto[0:150],lw=linewid,color=myyell)
			#for el in auto: print(el)

			ax10.set_title('rad hist',loc='left',fontsize=8,color=mybritegrn)
			ax10.set_xlabel('rad',fontsize=8,color=mybritegrn)
			ax10.set_ylabel('# in bin',fontsize=8,color=mybritegrn)			
		
			#ax11.set_title('ang vs k last 1/10',loc='left',fontsize=8,color=mybritegrn)
			#ax11.plot(angdata[end4:],lw=linewid,color=myteal)

			ax12.set_title('ang hist',loc='left',fontsize=8,color=mybritegrn)
			ax12.set_xlabel('ang',fontsize=8,color=mybritegrn)
			ax12.set_ylabel('# in bin',fontsize=8,color=mybritegrn)			

			ax0.set_xlim(xmin,xmax);ax0.set_ylim(ymin,ymax)
			ax5.set_xlim(0,kappa-trimend);ax5.set_ylim(radmin,radmax)
			ax7.set_xlim(0,kappa-trimend);ax7.set_ylim(angmin,angmax)

			if not doani:
				ax0.plot(xdata[0:kappa-trimend],ydata[0:kappa-trimend],lw=linewid,color=myteal)
				ax5.plot(raddata[0:kappa-trimend],lw=linewid,color=myturq)
				ax7.plot(angdata[0:kappa-trimend],lw=linewid,color=myteal)
				ax10.hist(raddata[0:kappa-trimend],bins=numbins,normed=True,color=myyell)
				ax12.hist(angdata[0:kappa-trimend],bins=numbins,normed=True,color=myyell2)				
				
			#___________________________________________________________________#
			data=raddata
			datalbl='rad'
			maxfreq=int(kappa/15)
			Fs = 300  # sampling rate - HIGHER = SHARPER PEAK
			Ts = 1.0/Fs # sampling interval
			n=kappa-2 # length of the signal
			kk = np.arange(kappa-2)
			T = kappa/Fs
			frq2 = kk/T # two sides frequency range
			frq1 = frq2[1:int(n/2)] # one side frequency slice
			Y2 = 2*np.fft.fft(data)/n # fft computing and norm
			Y1 = Y2[1:int(n/2)]
			ax6.set_title('FT('+datalbl+') vs period   Fsamp=300',loc='left',fontsize=8,color=mybritegrn)
			ax6.set_xlabel('period',fontsize=8,color=mybritegrn)
			ax6.set_ylabel('FT('+datalbl+')',fontsize=8,color=mybritegrn)
			ax6.plot(frq1[1:maxfreq],abs(Y1[1:maxfreq]),lw=linewid,color=myorange)

			data=angdata
			datalbl='ang'
			maxfreq=int(kappa/15)
			Fs = 300  # sampling rate - HIGHER = SHARPER PEAK
			Ts = 1.0/Fs # sampling interval
			n=kappa-2 # length of the signal
			kk = np.arange(kappa-2)
			T = kappa/Fs
			frq2 = kk/T # two sides frequency range
			frq1 = frq2[1:int(n/2)] # one side frequency slice
			Y2 = 2*np.fft.fft(data)/n # fft computing and norm
			Y1 = Y2[1:int(n/2)]
			ax8.set_title('FT('+datalbl+') vs period   Fsamp=300',loc='left',fontsize=8,color=mybritegrn)
			ax8.set_xlabel('period',fontsize=8,color=mybritegrn)
			ax8.set_ylabel('FT('+datalbl+')',fontsize=8,color=mybritegrn)
			ax8.plot(frq1[1:maxfreq],abs(Y1[1:maxfreq]),lw=linewid,color=myorange2)			

			data=auto[0:300]
			datalbl='power spectrum(rad)'
			maxfreq=int(kappa/15)
			Fs = 300  # sampling rate - HIGHER = SHARPER PEAK
			Ts = 1.0/Fs # sampling interval
			n=kappa-2 # length of the signal
			kk = np.arange(kappa-2)
			T = kappa/Fs
			frq2 = kk/T # two sides frequency range
			frq1 = frq2[1:int(n/2)] # one side frequency slice
			Y2 = 2*np.fft.fft(data)/n # fft computing and norm
			Y1 = Y2[1:int(n/2)]
			ax11.set_title('FT('+datalbl+') vs period   Fsamp=300',loc='left',fontsize=8,color=mybritegrn)
			ax11.set_xlabel('period',fontsize=8,color=mybritegrn)
			ax11.set_ylabel('FT('+datalbl+')',fontsize=8,color=mybritegrn)
			ax11.plot(frq1[1:maxfreq],abs(Y1[1:maxfreq]),lw=linewid,color=myyell2)	
			#___________________________________________________________________#

			plt.show(block=False);fig.canvas.draw()
		
			if doani:

				# The -1 in xdata[n*lag-1:(n+1)*lag] is needed to make the line from the slice start from the last point in the previous slice
				# so the orbit is connected which can be seen by print(n,n*lag-1,(n+1)*lag,xdata[n*lag-1:(n+1)*lag])
				# But for the first slice this results in the slice [-1:0] which means it tries to start at the last element "thru" the first
				# so the first slice is plotted manually before the loop.
				#
				# At the last slice, lag=int(kappa/numgrads) may not divide evenly into kappa
				# so there will typically be a few points at the end not covered by the loop;
				# these are handled manually after the loop.

				#print(xdata)
				ax0.set_xlim(xmin,xmax);ax0.set_ylim(ymin,ymax)
				ax5.set_xlim(0,kappa-trimend);ax5.set_ylim(radmin,radmax)
				ax7.set_xlim(0,kappa-trimend);ax7.set_ylim(angmin,angmax)
				ax10.set_xlim(0,radmax);ax12.set_xlim(0,angmax)

				n=0
				#print('\nFIRST SLICE') # handle first slice manually
				#print('n,n*lag,(n+1)*lag,xdata[n*lag:(n+1)*lag]')
				#print(n,n*lag,(n+1)*lag,xdata[n*lag:(n+1)*lag])					
				ax0.plot(xdata[n*lag:(n+1)*lag],ydata[n*lag:(n+1)*lag],lw=linewid,color=alumen[n])
				ax5.plot(raddata[n*lag:(n+1)*lag],lw=linewid,color=alumen[n])
				ax7.plot(angdata[n*lag:(n+1)*lag],lw=linewid,color=alumen[n])
				ax10.hist(raddata[0:(n+1)*lag],bins=numbins,normed=True,color=myyell)
				ax12.hist(angdata[0:(n+1)*lag],bins=numbins,normed=True,color=myyell2)

				plt.show(block=False);fig.canvas.draw()
				time.sleep(linesleep)

				#for n in range(1,numgrads):
				while (n+1)*lag<kappa-lag:
					n+=1
					#print('n,n*lag-1,(n+1)*lag,xdata[n*lag-1:(n+1)*lag]')
					#print(n,n*lag-1,(n+1)*lag,xdata[n*lag-1:(n+1)*lag])

					ax0.plot(xdata[n*lag-1:(n+1)*lag],ydata[n*lag-1:(n+1)*lag],lw=linewid,color=alumen[n])
					
					refreshone(self,5);refreshone(self,7);refreshone(self,10);refreshone(self,12)						
					ax5.plot(raddata[0:(n+1)*lag],lw=linewid,color=alumen[n])
					ax7.plot(angdata[0:(n+1)*lag],lw=linewid,color=alumen[n])
					ax10.hist(raddata[0:(n+1)*lag],bins=numbins,normed=True,color=myyell)
					ax12.hist(angdata[0:(n+1)*lag],bins=numbins,normed=True,color=myyell2)

					plt.show(block=False);fig.canvas.draw()
					time.sleep(linesleep)

				n=numgrads-1

			nice.append(params);xnice.append(params[1]);ynice.append(params[2])
			intfile=str(kappa)+'_'+str(p)+'_'+str(q)+'.txt'

			if dosave: 
				# save to DB if it exists
				try:
					cnx = mysql.connector.connect(user='root', password='YOURMYSQLPW', host='127.0.0.1', database='mandel')
					cursor = cnx.cursor()
					add_orbit = 'INSERT INTO orbit (p,q,kappa,maxrad) ' + 'VALUES ('+str(p)+','+str(q)+','+str(kappa)+','+str(maxrad)+')'
					cursor.execute(add_orbit)
					cnx.commit()
					cursor.close()
					cnx.close()
				#else save to csv
				except:
					if os.path.isdir('imgs'):intfile='imgs/'+intfile
					intdata=''
					for el in nice:
						csvrow=str(el[0])+','+str(el[1])+','+str(el[2])+'\n'
						intdata+=str(csvrow)
					with open(intfile,'w') as intfile:intfile.write(intdata)
				# save png
				outfile=str(kappa)+'_'+str(p)+'_'+str(q)
				if os.path.isdir('imgs'):outfile='imgs/'+outfile
				outpic=outfile+'.png'; savefig(outpic,facecolor=fig.get_facecolor(),bbox_inches='tight')

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

doani=False 	# animate the orbit, rad, ang and histogram plots
doloop=True		# loop thru random orbits, not just one
dosave=True 	# save params to DB or file, and save png
doang=False 	# external angle, not implemented yet
boreme=False 	# pick a long orbit which escapes at the end

maxiters=5000	# max iterations
minbored=1000	# minimum non-boring orbit iterations
maxrad=2.0		# defines the escape criterion
trimend=4		# omits the final few iterations from some of the plots
numbins=200		# no. of bins in the histograms

numgrads=3 		# how many slices to plot the animated orbit
linewid=.6		# line width

figclose=False 	# close after plotting
figsleep=0		# various sleep intervals
finalsleep=0
linesleep=0

wid=1000;ht=700;xpos=10;ypos=100	# window size & posn
wid=1200;ht=650;xpos=2100;ypos=100
wid=1800;ht=1100;xpos=0;ypos=0

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
	cx=-0.472262157722;cy=-0.539097310145

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
cx=0.2662964889352877;cy=0.004143476663879161
rrp=.025;rrn=-.025
rrrx=np.random.uniform(rrn,rrp);rrry=np.random.uniform(rrn,rrp)
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

# MULTIPLE FLUTES
cx=-0.660523088721;cy=0.33339541171

# OTHER BUDS:
2 -0.75 1.22464679915e-16
3 -0.125 0.649519052838
4 0.25 0.5
5 0.356762745781 0.328581945074
6 0.375 0.216506350946
7 0.367375134418 0.147183763189
8 0.353553390593 0.103553390593
9 0.339610177143 0.0751918665902
10 0.327254248594 0.0561284970724
11 0.316773013165 0.0429124098892
12 0.308012701892 0.0334936490539
13 0.300711826144 0.0266156195485
14 0.294611983487 0.0214839989418
15 0.289490077232 0.0175821151686
16 0.285163070959 0.0145650208859
17 0.281483885397 0.0121969221819
18 0.278335199613 0.0103131692412
19 0.275623493501 0.00879655642992
20 0.273274009554 0.00756218411436
21 0.271226709314 0.00654757268955
22 0.269433103599 0.00570607405682
23 0.267853792537 0.00500239806965
24 0.266456562198 0.00440952255126
25 0.265214910553 0.003906525057
26 0.2641069023 0.00347703913284
27 0.263114275209 0.0031081403211
28 0.262221739115 0.00278953219877
29 0.261416422937 0.00251294471545
30 0.260687435956 0.00227168463993
31 0.260025517721 0.002060296266
32 0.259422757074 0.00187430291679
33 0.258872365377 0.00171000826512
34 0.258368492491 0.0015643423615
35 0.257906076639 0.00143474137898
36 0.25748072131 0.00131905300205
37 0.257088593809 0.00121546147541
38 0.256726341276 0.0011224278392
39 0.256391020832 0.00103864197851
40 0.256080041224 0.000962983926379
41 0.255791113876 0.000894492460639
42 0.255522211666 0.000832339485361
43 0.25527153406 0.000775809026193


# SCATTER PLOT (MANDELBROT MAP)
#ax1.set_title('xmap-ymap',fontsize=8,color=mybritegrn)
#ax1.scatter(xnice,ynice,s=1,lw=linewid,color=myorange)

# INTERNAL ANGLE
# if doang...
#ax1111.set_title('ang2int-k',fontsize=8,color=mybritegrn)
#ax1111.plot(ang2data,lw=linewid,color=myturq)

"""