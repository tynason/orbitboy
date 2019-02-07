#!/usr/bin/python3
#
# orbitboy.py
# ted nason 2018
# https://github.com/tynason/orbitboy
#___________________________________________________________________#
import matplotlib as mpl
from pylab import *
import scipy.fftpack
import numpy as np
import random
import datetime
import time
import os
import mysql.connector
#___________________________________________________________________#

class Mandel(object):
	"Mandelbrot orbit plotting"
	def __init__(self,doani,doloop,dosave,doang,angerror,dodata,chunk,chunksleep,maxiters,minbored,boreme,maxrad,trimend,numbins,numgrads,figclose,figsleep,finalsleep,linesleep,wid,ht,xpos,ypos):
		self.doani=doani
		self.doloop=doloop
		self.dosave=dosave
		self.doang=doang
		self.angerror=angerror
		self.dodata=dodata
		self.chunk=chunk
		self.chunksleep=chunksleep
		self.maxiters=maxiters
		self.minbored=minbored
		self.boreme=boreme
		self.maxrad=maxrad
		self.trimend=trimend
		self.numbins=numbins
		self.numgrads=numgrads
		self.figclose=figclose
		self.figsleep=figsleep
		self.finalsleep=finalsleep
		self.linesleep=linesleep
		self.wid=wid
		self.ht=ht
		self.xpos=xpos
		self.ypos=ypos

	# https://matplotlib.org/users/customizing.html
	mpl.rcParams['font.size']=7
	mpl.rcParams['axes.labelsize']=7
	mpl.rcParams['xtick.color']='#aaaaaa'
	mpl.rcParams['ytick.color']='#aaaaaa'
	mpl.rcParams['lines.linewidth']=0.4

	def getcolor(self):
		brite=0
		while True:
			r=np.random.uniform(0.4,0.8)
			g=np.random.uniform(0.0,0.2)
			b=np.random.uniform(0.4,0.8)
			# not too dark
			if r>0.6 or b>0.6: break
		print('rgbfore: ',int(r*256),int(g*256),int(b*256))
		return (r,g,b)

	def lumengen(self,fore,back,grads): # get colors between start (back) and end (fore) colors
		rangeRGB=[x1-x2 for (x1,x2) in zip(fore,back)] # the range of RGB[0-1] to be covered
		segRGB=list(map(lambda x: x/grads,rangeRGB)) # the amount to decrement each element RGB
		R=np.zeros((grads,));G=np.zeros((grads,));B=np.zeros((grads,)) # start w/fore and decrement to back
		for nn in range(self.numgrads): R[nn]=fore[0]-nn*segRGB[0];G[nn]=fore[1]-nn*segRGB[1];B[nn]=fore[2]-nn*segRGB[2]
		return list(zip(R,G,B))

	def datagen(self,maxiterrs):
		print('searching...')
		boredcount=0 
		while True:
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
			elif flip==6: # BULB AT (-1.0)
				theta=np.random.uniform(0,2*math.pi);rrr=np.random.uniform(0,0.05)
				cx=(0.25+rrr)*cos(theta)-1;cy=(0.25+rrr)*sin(theta)
			#___________________________________________________________________#

			xdata.append(cx);ydata.append(cy)
			self.mandfunc='z*z+c'

			k=0;c=complex(cx,cy);z=complex(0.0,0.0)
			while abs(z)<self.maxrad and k<maxiterrs:
				k+=1;z=z*z+c
				xdata.append(z.real);ydata.append(z.imag)
			
			if self.boreme:
				if k>=self.minbored: # it's boring but over the minbored, so stop looking
					print('BORING #',boredcount,'kappa=',k,'\tcx=',cx,'\tcy=',cy)
					break # go on to calculate stats and return
				else: continue # it's less then minbored so keep looking
			else:
				if k>self.minbored and k<maxiterrs: # we found a non-boring one
					break # go on to calculate stats and return
				else: continue # keep looking

		for n in range(0,k-2):
			rad=xdata[n]**2+ydata[n]**2
			raddata.append(rad)

			# external angle
			angerror='none'	
			if abs(xdata[n])<1e-8:
				ang=90 # no arctan of 0, so just fix it and don't break
				angerror='ext_angle arctan(0) set ang=90'
				print('QUASI-EXCEPTION angerror: ', angerror)
			else:
				ang=math.atan2(ydata[n],xdata[n]);ang=math.degrees(ang)%360
			angdata.append(ang)

			
			# internal angle
			# math seems wrong here, need to revisit
			# yet same routine works fine with spiro
			if self.doang:
				for n in range(0,k-2):
					point1=xdata[n],ydata[n];point2=xdata[n+1],ydata[n+1];point3=xdata[n+2],ydata[n+2]
					lineA=([point1[0],point1[1]]),([point2[0],point2[1]]);lineB=point2,point3
					vA=([(lineA[0][0]-lineA[1][0]),(lineA[0][1]-lineA[1][1])])
					vB=([(lineB[0][0]-lineB[1][0]),(lineB[0][1]-lineB[1][1])])
					dot_prod=dot(vA,vB);magA=dot(vA,vA)**0.5;magB=dot(vB,vB)**0.5

					if abs(magA)<1e-8:
						self.angerror='magA=0'
						print('EXCEPTION angerror: ', self.angerror); break
					if abs(magB)<1e-8:
						self.angerror='magB=0'
						print('EXCEPTION angerror: ', self.angerror); break
					argg=dot_prod/magB/magA
					if abs(argg)>1:
						self.angerror='arccos(arg>1)'
						print('EXCEPTION angerror: ', self.angerror); break

					ang=math.acos(argg)
					ang_deg=180-math.degrees(ang)%360
					ang2data.append(ang_deg)

				ang2data.pop(0)

		params=[k,cx,cy,self.angerror]
		xmin=min(xdata[:k-self.trimend]);xmax=max(xdata[:k-self.trimend]);xavg=mean(xdata);xdev=std(xdata);xstats=[xmin,xmax,xavg,xdev]
		ymin=min(ydata[:k-self.trimend]);ymax=max(ydata[:k-self.trimend]);yavg=mean(ydata);ydev=std(ydata);ystats=[ymin,ymax,yavg,ydev]
		radmin=min(raddata);radmax=max(raddata);radavg=mean(raddata);raddev=std(raddata);radstats=[radmin,radmax,radavg,raddev]
		angmin=min(angdata);angmax=max(angdata);angavg=mean(angdata);angdev=std(angdata);angstats=[angmin,angmax,angavg,angdev]
		return params,xstats,ystats,radstats,angstats,xdata,ydata,raddata,angdata,boredcount,ang2data

	def plotme(self):

		def refreshall(self):
			for ax in fig.axes:
				currcolor = ax.patch.get_facecolor();currtitle = ax.get_title(loc='left')
				currxlabel=ax.get_xlabel();currylabel=ax.get_ylabel()
				currxlim=ax.get_xlim();currylim=ax.get_ylim()
				ax.clear();ax.patch.set_facecolor(currcolor)
				ax.set_title(currtitle,loc='left',color=mybritegrn)
				ax.grid(False);ax.patch.set_alpha(1.0)
				ax.set_xlabel(currxlabel,color=mybritegrn)
				ax.set_ylabel(currylabel,color=mybritegrn)
				#ax.set_xlim(currxlim);ax.set_ylim(currylim)

		def refreshone(self,n):
			ax=fig.axes[n]
			currcolor = ax.patch.get_facecolor();currtitle = ax.get_title(loc='left')
			currxlabel=ax.get_xlabel();currylabel=ax.get_ylabel()
			currxlim=ax.get_xlim();currylim=ax.get_ylim()
			ax.clear();ax.patch.set_facecolor(currcolor)
			ax.set_title(currtitle,loc='left',color=mybritegrn)
			ax.grid(False);ax.patch.set_alpha(1.0)
			ax.set_xlabel(currxlabel,color=mybritegrn)
			ax.set_ylabel(currylabel,color=mybritegrn)
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





		win = plt.gcf().canvas.manager.window
		fig.canvas.manager.window.wm_geometry('%dx%d%+d%+d' % (self.wid,self.ht,self.xpos,self.ypos))
		fig.patch.set_facecolor(rgbback)
		plt.tight_layout(pad=0.1,w_pad=0.1,h_pad=0.5)
		fig.subplots_adjust(top=0.935)





		nice=[];xnice=[];ynice=[] #list of non-boring orbits

		while True:
			start_time=time.time()

			# deal with colors here
			# lumon is the end color of the plot gradient
			lumon=(255,20,200) # your standard fuscia
			lumon=list(map(lambda x: x/256,lumon))

			#randomize the end color here
			resultt=self.getcolor()
			lumon=resultt

			# lumoff is the start color of the plot gradient
			lumoff=(0,225,125)

			lumoff=list(map(lambda x: x/256,lumoff))
			lumen=self.lumengen(lumon,lumoff,self.numgrads)
			alumen=lumen[::-1]

			# get the data for an orbit
			result=self.datagen(self.maxiters)

			# set names for result of datagen here
			params=result[0]
			kappa=params[0];p=params[1];q=params[2];boredcount=result[9];angerr=params[3]

			# if numgrads>kappa, you are trying to plot more slices than there are data points so set lag=1
			if kappa>self.numgrads:
				lag=int(kappa/self.numgrads)
			else:
				lag=1

			xdata=result[5];ydata=result[6];raddata=result[7];angdata=result[8]
			xmin=result[1][0];xmax=result[1][1];ymin=result[2][0];ymax=result[2][1]
			radmin=result[3][0];radmax=result[3][1]
			angmin=result[4][0];angmax=result[4][1]
			ang2data=result[10]
			radavg=result[3][2];raddev=result[3][3]
			angavg=result[4][2];angdev=result[4][3]
			#___________________________________________________________________#

			# print some stuff
			if self.dodata: # full orbit stats print
				###################### PRINT HEADER ######################
				print('\n')
				myheader = ['kappa','n','xdata[n]','ydata[n]']
				for item in myheader: print('{: <22}'.format(item),end='')
				print('\n',end='')
				print('+'*100)

				###################### PRINT DATA IN CHUNKS ######################
				start=0
				end=self.chunk
				while end<=kappa:
					for item in myheader: print('{: <22}'.format(item),end='')
					print('\n',end='')
					print('+'*100)

					for n in range(start,end):
						mydata = [kappa,n,xdata[n],ydata[n]]
						for item in mydata: print('{: <22}'.format(item),end='')
						print('\n',end='')
						time.sleep(self.linesleep)

					if end==kappa:break
					time.sleep(self.chunksleep)
					start+=self.chunk
					end+=self.chunk
					if end>kappa: end=kappa
					print('+'*100)
				time.sleep(self.chunksleep)

			print('\nSELECTED non-boring p,q after',str(boredcount),'tries:')

			# vertical params print
			print('\tkappa:',kappa)
			print('\tp='+str(p))
			print('\tq='+str(q))

			print('\tboreme:',self.boreme)
			print('\tminbored:',self.minbored)
			print('\tmaxiters',self.maxiters)

			print('\tradavg:',radavg)
			print('\traddev:',raddev)
			print('\tangavg:',angavg)
			print('\tangdev:',angdev)
			print('\tangerr:',angerr)

			print('\tnumbins:',self.numbins)
			print('\tnumgrads',self.numgrads)
			print('\tlag:',lag)
			print('\ttrimend:',self.trimend)

			"""
			# horizontal params print
			myheader = ['kappa','p','q','maxrad','maxiters','radavg','raddev','angavg','angdev','angerror']
			for item in myheader: print('{: <22}'.format(item),end='')
			print('\n')
			mydata = [kappa,p,q,self.maxrad,self.maxiters,radavg,raddev,angavg,angdev,self.angerror]
			for item in mydata: print('{: <22}'.format(item),end='')
			print('\n')

			# horizontal params print
			myheader = ['boreme','minbored','numbins','numgrads','lag','trimend']
			for item in myheader: print('{: <22}'.format(item),end='')
			print('\n')
			mydata = [self.boreme,self.minbored,self.numbins,self.numgrads,lag,self.trimend]
			for item in mydata: print('{: <22}'.format(item),end='')
			print('\n')
			"""
			#___________________________________________________________________#

			# plot some stuff
			mytitle='Mandelbrot ' +self.mandfunc+ '  p= '+str(p)+'  q= '+str(q)+'  kappa= '+str(kappa)+'  maxrad= '+str(self.maxrad)
			fig.suptitle(mytitle,fontsize=9,color='#00ff00')
			refreshall(self)
			
			end1=int(kappa/10);end2=int(kappa/2);end3=int(kappa*0.9)
			lumend1=alumen[int(self.numgrads/10)]
			lumend2=alumen[int(self.numgrads/2)]
			lumend3=alumen[int(self.numgrads*0.9)]
			lumend4=alumen[int(self.numgrads)-1]

			ax0.set_title('x-y orbit plot  trimend='+str(self.trimend),loc='left',color=mybritegrn)
			ax0.set_xlim(xmin,xmax);ax0.set_ylim(ymin,ymax)

			ax1.set_title('first 1/10',loc='left',color=mybritegrn)
			ax1.plot(xdata[0:end1],ydata[0:end1],color=lumend1)
				
			ax2.set_title('1/10 - 1/2',loc='left',color=mybritegrn)
			ax2.plot(xdata[end1:end2],ydata[end1:end2],color=lumend2)
			
			ax3.set_title('1/2 - 90%',loc='left',color=mybritegrn)
			ax3.plot(xdata[end2:end3],ydata[end2:end3],color=lumend3)
			
			#ax4.set_xlim(xmin-0.8*abs(xmin),xmax+0.8*abs(xmax))
			#ax4.set_ylim(ymin-0.8*abs(ymin),ymax+0.8*abs(ymax))
			ax4.set_title('last 1/10  trimend=0',loc='left',color=mybritegrn)
			ax4.plot(xdata[end3:],ydata[end3:],color=lumend4)

			ax5.set_title('rad vs k',loc='left',color=mybritegrn)
			ax7.set_title('ang vs k',loc='left',color=mybritegrn)

			ax9.set_title('rad vs k last 1/10',loc='left',color=mybritegrn)
			ax9.plot(raddata[end3:],color=myorange)

			ax6.set_title('rad hist',loc='left',color=mybritegrn)
			ax6.set_xlabel('rad',color=mybritegrn)
			ax6.set_ylabel('# in bin',color=mybritegrn)			

			ax11.set_title('ang vs k last 1/10',loc='left',color=mybritegrn)
			ax11.plot(angdata[end3:],color=myorange2)
			
			ax8.set_title('ang hist',loc='left',color=mybritegrn)
			ax8.set_xlabel('ang',color=mybritegrn)
			ax8.set_ylabel('# in bin',color=mybritegrn)			

			ax5.set_xlim(0,kappa-self.trimend);ax5.set_ylim(radmin,radmax)
			ax7.set_xlim(0,kappa-self.trimend);ax7.set_ylim(angmin,angmax)

			ax6.hist(raddata[0:kappa-self.trimend],bins=self.numbins,normed=True,color=myyell)
			ax8.hist(angdata[0:kappa-self.trimend],bins=self.numbins,normed=True,color=myyell2)
			#___________________________________________________________________#

			# do some fourier stuff
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

			ax10.set_title('FT('+datalbl+') vs period   Fsamp=300',loc='left',color=mybritegrn)
			ax10.set_xlabel('period',color=mybritegrn)
			ax10.set_ylabel('FT('+datalbl+')',color=mybritegrn)
			ax10.plot(frq1[1:maxfreq],abs(Y1[1:maxfreq]),color=myorange)

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

			ax12.set_title('FT('+datalbl+') vs period   Fsamp=300',loc='left',color=mybritegrn)
			ax12.set_xlabel('period',color=mybritegrn)
			ax12.set_ylabel('FT('+datalbl+')',color=mybritegrn)
			ax12.plot(frq1[1:maxfreq],abs(Y1[1:maxfreq]),color=myorange2)
			#___________________________________________________________________#

			if not self.doani:
				ax0.plot(xdata[0:kappa-self.trimend],ydata[0:kappa-self.trimend],color=lumend4)
				ax5.plot(raddata[0:kappa-self.trimend],color=myorange)
				ax7.plot(angdata[0:kappa-self.trimend],color=myorange2)
				plt.show(block=False);fig.canvas.draw()

			else:
				# homebrew quasi-animation
				#
				# The -1 in xdata[n*lag-1:(n+1)*lag] is needed to make the line from the slice start from the last point in the previous slice
				# so the orbit is connected which can be seen by print(n,n*lag-1,(n+1)*lag,xdata[n*lag-1:(n+1)*lag])
				# But for the first slice this results in the slice [-1:0] which means it tries to start at the last element "thru" the first
				# so the first slice is plotted manually before the loop.
				#
				# At the last slice, lag=int(kappa/self.numgrads) may not divide evenly into kappa
				# so there will typically be a few points at the end not covered by the loop;
				# these are handled manually after the loop.

				refreshone(self,5);refreshone(self,7)
				ax5.set_xlim(0,kappa-self.trimend);ax5.set_ylim(radmin,radmax)
				ax7.set_xlim(0,kappa-self.trimend);ax7.set_ylim(angmin,angmax)

				try:
					n=0
					#print('\nFIRST SLICE') # handle first slice manually
					#print('n,n*lag,(n+1)*lag,xdata[n*lag:(n+1)*lag]')
					#print(n,n*lag,(n+1)*lag,xdata[n*lag:(n+1)*lag])

					ax0.plot(xdata[n*lag:(n+1)*lag],ydata[n*lag:(n+1)*lag],color=alumen[n])

					#ax5.plot(raddata[n*lag:(n+1)*lag],color=alumen[n])
					xx=np.arange(n*lag,(n+1)*lag)
					yy=np.array(raddata[n*lag:(n+1)*lag])
					ax5.plot(xx,yy,color=alumen[n])

					#ax7.plot(angdata[n*lag:(n+1)*lag],color=alumen[n])
					xx=np.arange(n*lag,(n+1)*lag)
					yy=np.array(angdata[n*lag:(n+1)*lag])
					ax7.plot(xx,yy,color=alumen[n])

					# histograms
					#refreshone(self,10);refreshone(self,12)
					refreshone(self,6);refreshone(self,8)
					ax6.set_xlim(0,radmax);ax8.set_xlim(0,angmax)
					ax6.hist(raddata[0:(n+1)*lag],bins=self.numbins,normed=True,color=alumen[n])
					ax8.hist(angdata[0:(n+1)*lag],bins=self.numbins,normed=True,color=alumen[n])

					plt.show(block=False);fig.canvas.draw()
					time.sleep(self.linesleep)
				except:
					pass

				#for n in range(1,self.numgrads):
				while (n+1)*lag<kappa-lag:
					try:
						n+=1
						#print('n,n*lag-1,(n+1)*lag,xdata[n*lag-1:(n+1)*lag]')
						#print(n,n*lag-1,(n+1)*lag,xdata[n*lag-1:(n+1)*lag])

						ax0.plot(xdata[n*lag-1:(n+1)*lag],ydata[n*lag-1:(n+1)*lag],color=alumen[n])
						
						yy=np.array(raddata[n*lag:(n+1)*lag])
						xx=np.arange(n*lag,n*lag+len(yy))
						ax5.plot(xx,yy,color=alumen[n])

						yy=np.array(angdata[n*lag-1:(n+1)*lag])
						xx=np.arange(n*lag,n*lag+len(yy))
						ax7.plot(xx,yy,color=alumen[n])

						# histograms
						#refreshone(self,10);refreshone(self,12)
						refreshone(self,6);refreshone(self,8)
						ax6.set_xlim(0,radmax);ax8.set_xlim(0,angmax)
						ax6.hist(raddata[0:(n+1)*lag],bins=self.numbins,normed=True,color=alumen[n])
						ax8.hist(angdata[0:(n+1)*lag],bins=self.numbins,normed=True,color=alumen[n])

						plt.show(block=False);fig.canvas.draw()
						time.sleep(self.linesleep)
					except:
						continue

				n=self.numgrads-1


			nice.append(params);xnice.append(params[1]);ynice.append(params[2])
			intfile=str(kappa)+'_'+str(p)+'_'+str(q)+'.txt'

			if self.dosave: 
				# save to DB if it exists
				try:
					cnx = mysql.connector.connect(user='root', password='YOURMYSQLPASSWORD', host='127.0.0.1', database='mandel')
					cursor = cnx.cursor()
					add_orbit = 'INSERT INTO orbit (p,q,kappa,maxrad,radavg,raddev,angavg,angdev) VALUES('+str(p)+','+str(q)+','+str(kappa)+','+str(self.maxrad)+','+str(radavg)+','+str(raddev)+','+str(angavg)+','+str(angdev)+')'
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
			time.sleep(self.figsleep)
			if not self.doloop: break

		if self.figclose:
			time.sleep(self.finalsleep)
			plt.close()
		else:
			plt.show(block=True)


mygunmet='#113344';myblue='#11aacc';mydkblue='#0000cc'
myturq='#00ffff';myturq2='#11bbbb';myteal='#00ffcc';myteal2='#00ccaa'
mygreen='#44bb44';mybritegrn='#00ff66';myfuscia='#ff00ff';myfuscia2='#dd0099'
mypurp='#ff00cc';mypurp2='#9933ff';myred='#cc0000';myorange='#ffaa00'
myorange2='#ff6600';myyell='#ffff00';myyell2='#ffcc00'

mygunmet2='#052529'
mygunmet2='#052529'

rgbback=mygunmet2

#___________________________________________________________________#

doani=True 		# animate the orbit, rad, and ang plots
doloop=True		# loop thru random orbits, not just one
dosave=False 	# save params to DB or file, and save png
doang=False 	# external angle, not implemented yet
angerror='none'

dodata=False
chunk=40
chunksleep=0

maxiters=24000	# max iterations
minbored=2400	# minimum non-boring orbit iterations

boreme=False 	#  False to pick a long orbit which escapes at the end
maxrad=2.0		# defines the escape criterion

trimend=4		# omits the final few iterations from some of the plots
numbins=120		# no. of bins in the histograms
numgrads=12		# how many slices to plot the animated orbit

figclose=False 	# close after plotting
figsleep=0		# various sleep intervals
finalsleep=0
linesleep=0

wid=1800;ht=1200;xpos=0;ypos=0




while True:
	mymand=Mandel(doani,doloop,dosave,doang,angerror,dodata,chunk,chunksleep,maxiters,minbored,boreme,maxrad,trimend,numbins,numgrads,figclose,figsleep,finalsleep,linesleep,wid,ht,xpos,ypos)
	print(mymand.__doc__)
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
cx=-0.735421537445;cy=-0.163709777921 how only 1 peak at 26 ?
cx=-0.7506095013167247;cy=0.02017840558087991
cx=-0.7451660763608009;cy=0.08098427487766885
cx=-0.748788372883687;cy=0.0692479509378728
cx=-0.5013475449887075;cy=0.516886282748629
cx=-0.0170723831006;cy=0.639306902532

# MULTIPLE FLUTES
cx=-0.660523088721;cy=0.33339541171

# starts out like 4, widens to circle
cx=-0.656444674147;cy=-0.341203474013

# starts out flutes, widens to circle
cx=-0.68943228887;cy=-0.277715066752

# 18 star
cx=-0.669024875506;cy=-0.350659130696

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
#ax1.set_title('xmap-ymap',color=mybritegrn)
#ax1.scatter(xnice,ynice,s=1,color=myorange)

# INTERNAL ANGLE
# if doang...
#ax1111.set_title('ang2int-k',color=mybritegrn)
#ax1111.plot(ang2data,color=myturq)


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

				if len(xx)!=len(yy):
					xx.pop()
					print('NOT EQUAL')
"""
