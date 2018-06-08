#subspifri.py
# tnason 2018
#___________________________________________________________________#
import matplotlib.pyplot as plt
from pylab import *
import random
import time
#___________________________________________________________________#

class Spiro(object):
	def __init__(self,boring,borangle,xpos,ypos,wid,ht,figsleep,finalsleep):
		pass

	def datagen(self,maxlines):
		xdata=[];ydata=[];xstats=[];ystats=[];params=[]
		continn=True
		angmin=100;bored=0
		while continn:
			RB=np.random.uniform(0,80);RS=np.random.uniform(0,30);OO=np.random.uniform(0,100)
			bored+=1;xdata=[];ydata=[]
			for tt in range(maxlines):
				xx=(RB+RS)*cos(tt)-(RS+OO)*cos(((RB+RS)/RS)*tt)
				yy=(RB+RS)*sin(tt)-(RS+OO)*sin(((RB+RS)/RS)*tt)
				xdata.append(xx);ydata.append(yy)
			angdata=[]
			for tt in range(maxlines-2):
				point1=xdata[tt],ydata[tt];point2=xdata[tt+1],ydata[tt+1];point3=xdata[tt+2],ydata[tt+2]
				lineA=([point1[0],point1[1]]),([point2[0],point2[1]]);lineB=point2,point3
				vA=([(lineA[0][0]-lineA[1][0]),(lineA[0][1]-lineA[1][1])])
				vB=([(lineB[0][0]-lineB[1][0]),(lineB[0][1]-lineB[1][1])])
				dot_prod=dot(vA,vB);magA=dot(vA,vA)**0.5;magB=dot(vB,vB)**0.5
				ang=math.acos(dot_prod/magB/magA);ang_deg=180-math.degrees(ang)%360;angdata.append(ang_deg)
			angmin=min(angdata[:]);params=[RB,RS,OO,angmin]
			if boring:continn=False #if boring, this test succeeds and the current params are accepted
			else:
				if angmin<borangle:continn=False # if not theloop continues or exits based on this

		xmin=min(xdata);xmax=max(xdata);ymin=min(ydata);ymax=max(ydata)
		xmean=mean(xdata);ymean=mean(ydata);xspread=xmax-xmin;yspread=ymax-ymin
		xstats=[xmin,xmax,xmean,xspread];ystats=[ymin,ymax,ymean,yspread]
		params=[RB,RS,OO,angmin]
		return xstats,ystats,xdata,ydata,params

	def plotme(self,figscale,maxlines,lag,doforward,doloop,doprint,doprintappend,doprintreveal,doprintall,linewid,rgbfore,rgbback,digits):
		start_time=time.time()
		fig=plt.figure()
		fig.canvas.manager.window.wm_geometry("%dx%d%+d%+d" % (wid,ht,xpos,ypos))
		fig.patch.set_facecolor(rgbback)
		win=plt.gcf().canvas.manager.window
		ax=gca();ax.patch.set_facecolor(rgbback)
		ax.grid(True);ax.patch.set_alpha(1.0)
		ax.tick_params(axis='x',labelsize=8,labelcolor='#ffffff')
		ax.tick_params(axis='y',labelsize=8,labelcolor='#ffffff')
		plt.tight_layout()
		
		def refresh(draw=False):
			axcolor=ax.patch.get_facecolor()
			ax.clear()
			ax.patch.set_facecolor(axcolor)
			ax.grid(True);ax.patch.set_alpha(1.0)
			ax.tick_params(axis='x',labelsize=8,labelcolor='#ffffff')
			ax.tick_params(axis='y',labelsize=8,labelcolor='#ffffff')
			ax.set_xlim(xmin,figscale*xmax);ax.set_ylim(figscale*ymin,ymax)
			#ax.set_xlim(figscale*xmin,figscale*xmax);ax.set_ylim(figscale*ymin,figscale*ymax)
			if draw: fig.canvas.draw()

		def printalldata(self):
			# slowww... all the data as vertical string
			catxxx=''.join(list(map(lambda x: str(x)+'\n', xdata))) #;print(catxxx)
			catyyy=''.join(list(map(lambda y: str(y)+'\n', ydata))) #;print(catyyy)
			myxboxxx=ax.text(0.89,0.98,catxxx,verticalalignment='top',horizontalalignment='right',transform=ax.transAxes,color='#00ff00',fontsize=7,alpha=1.0)
			myyboxxx=ax.text(0.99,0.98,catyyy,verticalalignment='top',horizontalalignment='right',transform=ax.transAxes,color='#00ff00',fontsize=7,alpha=1.0)

		def lumengen(self,fore,back,grads): # get colors between fore and back color for plotting
			rangeRGB=[x1 - x2 for (x1, x2) in zip(fore,back)] # the range of RGB[0-1] to be covered
			segRGB=list(map(lambda x: x/grads, rangeRGB)) # the amount to decrement each element RGB
			R=np.zeros((grads,));G=np.zeros((grads,));B=np.zeros((grads,)) # start w/fore and decrement to back
			#for nn in range(numgrads-1): # set the last value explicitly so only do the penultimate
			#R[numgrads-1]=rgbback[0];G[numgrads-1]=rgbback[1];B[numgrads-1]=rgbback[2]
			for nn in range(numgrads): R[nn]=fore[0]-nn*segRGB[0];G[nn]=fore[1]-nn*segRGB[1];B[nn]=fore[2]-nn*segRGB[2]
			return list(zip(R,G,B))

		contin=True # to tell if we should keep looping new spiros
		numiters=0 # the number of spiro plots looped thru
		numgrads=int(maxlines/lag) # the number of line batches for a single spiro plot
		
		while contin==True: # the main, outer loop generating a new spiro plot
			catapdxx='';catapdyy='';
			n=0;numiters+=1;catxx='';catyy='';start=[];end=[];myxslice=[];myyslice=[];myxtrunc=[];myytrunc=[]

			result=self.datagen(maxlines) # get data from class method

			xmin=result[0][0];xmax=result[0][1];xmeann=result[0][2];xspread=result[0][3]
			ymin=result[1][0];ymax=result[1][1];ymeann=result[1][2];yspread=result[1][3]
			xdata=result[2];ydata=result[3]
			RB=result[4][0];RS=result[4][1];OO=result[4][2]
			angmin=result[4][3]

			ax.set_xlim(xmin,figscale*xmax);ax.set_ylim(figscale*ymin,ymax)
			#ax.set_xlim(figscale*xmin,figscale*xmax);ax.set_ylim(figscale*ymin,figscale*ymax)
			if doprintall: printalldata(self);fig.canvas.draw();time.sleep(figsleep) # slow...
			lumen=lumengen(self,rgbfore,rgbback,numgrads)
			alumen=lumen[::-1]
			if not colorforward: lumen=alumen #print(lumen)
			
			start.append(0);end.append(lag+1)
			myxslice.append(xdata[0:lag+1]);myyslice.append(ydata[0:lag+1])
			myxtrunc.append(['%.4f' % elem for elem in myxslice[n]]);myytrunc.append(['%.4f' % elem for elem in myyslice[n]])
			#print('\n\nn, start([n], end[n], lumen[n]:');print(n,start[n],end[n],lumen[n]);print('\nmyxslice, myyslice');print(myxslice[n]);print(myyslice[n])
			for n in range(1,numgrads):
				start.append(start[n-1]+lag)
				end.append(end[n-1]+lag)
				myxslice.append(xdata[start[n]:end[n]])
				myyslice.append(ydata[start[n]:end[n]])
				if doprint:
					myxtrunc.append(['%.4f' % elem for elem in myxslice[n]])
					myytrunc.append(['%.4f' % elem for elem in myyslice[n]])

			# PLOT
			if doforward:
				reset=0
				for n in range(0,numgrads):
					ax.plot(myxslice[n],myyslice[n],linewidth=linewid,color=lumen[n],alpha=1.0)
					plt.show(False);fig.canvas.draw();time.sleep(figsleep)
					#print('\n\nn, start([n], end[n], lumen[n]:');print(n,start[n],end[n],lumen[n]);print('\nmyxslice, myyslice');print(myxslice[n]);print(myyslice[n])

					if doprint:
						catxx=''.join(list(map(lambda x: str(x)+' \n', myxtrunc[n]))) # the current slice text as vertical string, truncated
						catyy=''.join(list(map(lambda y: str(y)+'\n', myytrunc[n])))

						if doprintappend: # ditto, appended  #bbox=dict(facecolor=rgbfore, alpha=0.1)
							catapdxx+=catxx+'\n';catapdyy+=catyy+'\n';reset+=1;catxx=catapdxx;catyy=catapdyy							
							if reset*lag>40:catapdxx='';catapdyy='';reset=0		
						myxboxx=ax.text(0.88,0.98,catxx,horizontalalignment='right',verticalalignment='top',transform=ax.transAxes,color=lumen[n],fontsize=7)
						myyboxx=ax.text(0.98,0.98,catyy,horizontalalignment='right',verticalalignment='top',transform=ax.transAxes,color=lumen[n],fontsize=7)
						fig.canvas.draw();time.sleep(figsleep)
						
						if doprintreveal: # sequential reveal of current slice as vertical string
							xxx='';yyy=''
							for nn in range(0,lag+1):
								xxx+=str(myxtrunc[n][nn])+'\n'
								yyy+=str(myytrunc[n][nn])+'\n'
								myxbox=ax.text(0.10,0.98,xxx,verticalalignment='top',horizontalalignment='right',transform=ax.transAxes,color=lumen[n],fontsize=7)
								myybox=ax.text(0.20,0.98,yyy,verticalalignment='top',horizontalalignment='right',transform=ax.transAxes,color=lumen[n],fontsize=7)
								fig.canvas.draw();time.sleep(figsleep)
								if not numgrads-n==1: myxbox.remove();myybox.remove()
						
						if not numgrads-n==1: myxboxx.remove();myyboxx.remove()

			else:
				reset=0
				for n in range(numgrads-1,-1,-1):
					ax.plot(myxslice[n],myyslice[n],linewidth=linewid,color=lumen[n],alpha=1.0)
					plt.show(False);fig.canvas.draw();time.sleep(figsleep)

					if doprint:
						catxx=''.join(list(map(lambda x: str(x)+'\n', myxtrunc[n]))) # the current slice text as vertical string, truncated
						catyy=''.join(list(map(lambda y: str(y)+'\n', myytrunc[n])))

						if doprintappend: # ditto, appended  #bbox=dict(facecolor=rgbfore, alpha=0.1)
							catapdxx+=catxx+'\n';catapdyy+=catyy+'\n';reset+=1;catxx=catapdxx;catyy=catapdyy							
							if reset*lag>35:catapdxx='';catapdyy='';reset=0		
						myxboxx=ax.text(0.88,0.98,catxx,horizontalalignment='right',verticalalignment='top',transform=ax.transAxes,color=lumen[n],fontsize=7)
						myyboxx=ax.text(0.98,0.98,catyy,horizontalalignment='right',verticalalignment='top',transform=ax.transAxes,color=lumen[n],fontsize=7)
						fig.canvas.draw();time.sleep(figsleep)

						if doprintreveal: # sequential reveal of current slice as vertical string
							xxx='';yyy=''
							for nn in range(0,lag+1):
								xxx+=str(myxtrunc[n][nn])+'\n'
								yyy+=str(myytrunc[n][nn])+'\n'
								myxbox=ax.text(0.10,0.98,xxx,verticalalignment='top',horizontalalignment='right',transform=ax.transAxes,color=lumen[n],fontsize=7)
								myybox=ax.text(0.20,0.98,yyy,verticalalignment='top',horizontalalignment='right',transform=ax.transAxes,color=lumen[n],fontsize=7)
								fig.canvas.draw();time.sleep(figsleep)
								if not n==0: myxbox.remove();myybox.remove()

						if not n==0: myxboxx.remove();myyboxx.remove()

			time.time();elapsed=time.time()-start_time
			timerpt='iters: '+str(numiters)+'\nelapsed: '+str(elapsed)+'\navg secs: '+str(elapsed/numiters)+'\nangmin: '+str(angmin)+'\nmaxlines: '+str(maxlines)+'\nlag: '+str(lag)+'\ngrads: '+str(numgrads)+'\ndoprint: '+str(doprint)
			#print('\n'+timerpt)
			mytimebox=ax.text(0.02,0.02,timerpt,verticalalignment='bottom',horizontalalignment='left',transform=ax.transAxes,color='#00ff00',fontsize=7);fig.canvas.draw()
			if doloop==False: plt.show(block=True);contin=False
			time.sleep(finalsleep);refresh(True)
#___________________________________________________________________#
finalsleep=8;figsleep=0.0
wid=900;ht=750;xpos=30;ypos=30
boring=False;borangle=25;linewid=1.0
figscale=1.2;digits=4

lag=4
maxlines=105
doloop=True
doforward=False
colorforward=True

doprint=True
doprintreveal=True
doprintappend=True
doprintall=False

myblue=(17,170,204) #11aacc
myteal=(17,187,187) #11bbbb
myturq=(0,255,255) #00ffff
mygreen=(34,170,136) #22aa88
mybritegrn=(0,255,0) #00ff00
mypurp=(255,0,204) #ff00cc
mygunmet=(17,51,68) #113344

myblue=list(map(lambda x: x/256, myblue))
myteal=list(map(lambda x: x/256, myteal))
myturq=list(map(lambda x: x/256, myturq))
mygreen=list(map(lambda x: x/256, mygreen))
mybritegrn=list(map(lambda x: x/256, mybritegrn))
mypurp=list(map(lambda x: x/256, mypurp))
mygunmet=list(map(lambda x: x/256, mygunmet))

rgbfore=mybritegrn
rgbback=mygunmet
print(rgbfore,rgbback)

myspi1=Spiro(boring,borangle,xpos,ypos,wid,ht,figsleep,finalsleep)
myspi1.plotme(figscale,maxlines,lag,doforward,doloop,doprint,doprintappend,doprintreveal,doprintall,linewid,rgbfore,rgbback,digits)
#___________________________________________________________________#
