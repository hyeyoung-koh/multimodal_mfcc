i=0
for i in range(401,801):
    myfile='clip_'+str(i)
    mfcc_extraction('D:\\0401-0800\\0401-0800\\'+myfile,'wav')
    numpy.savetxt('D:\\0401-0800\\0401-0800\\'+myfile+'.csv',mfcc_feature,delimiter=",")
    i+=1

i=0
for i in range(801,1201):
    myfile='clip_'+str(i)
    mfcc_extraction('D:\\0801-1200\\0801-1200\\'+myfile,'wav')
    numpy.savetxt('D:\\0801-1200\\0801-1200\\'+myfile+'.csv',mfcc_feature,delimiter=",")
    i+=1

i=0
for i in range(1201,1601):
    myfile='clip_'+str(i)
    mfcc_extraction('D:\\1201-1600\\1201-1600\\'+myfile,'wav')
    numpy.savetxt('D:\\1201-1600\\1201-1600\\'+myfile+'.csv',mfcc_feature,delimiter=",")
    i+=1


i=0
for i in range(1601,2001):
    myfile='clip_'+str(i)
    mfcc_extraction('D:\\1601-2000\\1601-2000\\'+myfile,'wav')
    numpy.savetxt('D:\\1601-2000\\1601-2000\\'+myfile+'.csv',mfcc_feature,delimiter=",")
    i+=1

i=0
for i in range(2001,2401):
    myfile='clip_'+str(i)
    mfcc_extraction('D:\\2001-2400\\2001-2400\\'+myfile,'wav')
    numpy.savetxt('D:\\2001-2400\\2001-2400\\'+myfile+'.csv',mfcc_feature,delimiter=",")
    i+=1


i=0
for i in range(2401,2801):
    myfile='clip_'+str(i)
    mfcc_extraction('D:\\2401-2800\\2401-2800\\'+myfile,'wav')
    numpy.savetxt('D:\\2401-2800\\2401-2800\\'+myfile+'.csv',mfcc_feature,delimiter=",")
    i+=1

i=0
for i in range(2801,3201):
    myfile='clip_'+str(i)
    mfcc_extraction('D:\\2801-3200\\2801-3200\\'+myfile,'wav')
    numpy.savetxt('D:\\2801-3200\\2801-3200\\'+myfile+'.csv',mfcc_feature,delimiter=",")
    i+=1

i=0
for i in range(3201,3601):
    myfile='clip_'+str(i)
    mfcc_extraction('D:\\3201-3600\\3201-3600\\'+myfile,'wav')
    numpy.savetxt('D:\\3201-3600\\3201-3600\\'+myfile+'.csv',mfcc_feature,delimiter=",")
    i+=1

i=0
for i in range(3601,4001):
    myfile='clip_'+str(i)
    mfcc_extraction('D:\\3601-4000\\3601-4000\\'+myfile,'wav')
    numpy.savetxt('D:\\3601-4000\\3601-4000\\'+myfile+'.csv',mfcc_feature,delimiter=",")
    i+=1


i=0
for i in range(4001,4401):
    myfile='clip_'+str(i)
    mfcc_extraction('D:\\4001-4400\\4001-4400\\'+myfile,'wav')
    numpy.savetxt('D:\\4001-4400\\4001-4400\\'+myfile+'.csv',mfcc_feature,delimiter=",")
    i+=1

i=0
for i in range(4401,4801):
    myfile='clip_'+str(i)
    mfcc_extraction('D:\\4401-4800\\4401-4800-수정본\\'+myfile,'wav')
    numpy.savetxt('D:\\4401-4800\\4401-4800-수정본\\'+myfile+'.csv',mfcc_feature,delimiter=",")
    i+=1

i=0
for i in range(4801,5201):
    myfile='clip_'+str(i)
    mfcc_extraction('D:\\4801-5200\\4801-5200-수정본\\'+myfile,'wav')
    numpy.savetxt('D:\\4801-5200\\4801-5200-수정본\\'+myfile+'.csv',mfcc_feature,delimiter=",")
    i+=1


i=0
for i in range(5201,5601):
    myfile='clip_'+str(i)
    mfcc_extraction('D:\\5201-5600\\5201-5600-수정본\\'+myfile,'wav')
    numpy.savetxt('D:\\5201-5600\\5201-5600-수정본\\'+myfile+'.csv',mfcc_feature,delimiter=",")
    i+=1



