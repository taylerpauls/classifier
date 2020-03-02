import numpy as np
import os
import os.path
import matplotlib.pyplot as plt
import soundfile as sf
import librosa as lr
import pandas as pd

            
def get_zero_pad(location,dataset_file_train,dataset_file_test):
        smallest = 100
        dims_array = np.array([])
        train_files = []
        test_files = []
        count_test = 0
        count_tr = 0
        df_tr = pd.read_csv('/Users/I26259/wavenet-features/timit_train_files.txt',names="f")
        df_train = df_tr.as_matrix()
        df_te = pd.read_csv('/Users/I26259/wavenet-features/timit_test_files.txt',names="f")
        df_test = df_te.as_matrix()
        for dirpath, dirnames, filenames in os.walk(location):
            for filename in [f for f in filenames if f.endswith((".npz"))]:
                dataf = np.load(location+filename)
                dataf = np.array(dataf['pase'])
                second_dim = dataf.shape[1]
                dims_array = np.append(dims_array, second_dim)
                if second_dim > 599:
                    dataf = dataf[:,0:599]
                else:
                    dims = 599 - second_dim
                    pad = np.zeros((dims,100))
                    array = pad.reshape(100,dims)
                    dataf = np.concatenate((dataf,array),axis=1)
                fileID = filename.split(".")
                fileID = fileID[0]
                gender = filename[0]
                print(gender)
                dataf = dataf.astype('double')
                if gender == "m":
                    label = 0
                else:
                    label = 1
                entire_data = dataf,label,fileID
                if fileID in df_train:
                    count_tr += 1
                    train_files.append(entire_data)
                if fileID in df_test:
                    test_files.append(entire_data)
                    count_test+=1
                #print(len(processed_files))
        print(count_tr,count_test)
        #np.savez(dataset_file_train,*train_files)
        #np.savez(dataset_file_test,*test_files)
            
        
        return dims_array
        
def get_blocks(location,dataset_file_train,dataset_file_test):
    smallest = 1000
    dims_array = np.array([])
    train_files = []
    test_files = []
    count_test = 0
    count_tr = 0
    df_tr = pd.read_csv('/Users/I26259/wavenet-features/timit_train_files.txt',names="f")
    df_train = df_tr.as_matrix()
    df_te = pd.read_csv('/Users/I26259/wavenet-features/timit_test_files.txt',names="f")
    df_test = df_te.as_matrix()
    for dirpath, dirnames, filenames in os.walk(location):
        for filename in [f for f in filenames if f.endswith((".npy"))]:
            dataf = np.load(location+filename)
            dataf = np.array(dataf['pase'])
            second_dim = dataf.shape[1]
            dims_array = np.append(dims_array, second_dim)
            if second_dim < smallest:
                dataf = dataf[:,0:599]
                smallest = second_dim
                print(smallest)
    block_size = 113
    for dirpath, dirnames, filenames in os.walk(location):
        for filename in [f for f in filenames if f.endswith((".npz"))]:
            dataf = np.load(location+filename)
            dataf = np.array(dataf['pase'])
            second_dim = dataf.shape[1]
            dims_array = np.append(dims_array, second_dim)
            block_count = int(second_dim/block_size)
            start = 0
            stop = block_size-1
            count_block = 0
            for i in range(block_count):
                data_block = dataf[:,start:stop]
                start = stop+1
                stop = stop+block_size
                fileID = filename.split(".")
                fileID = fileID[0]
                gender = filename[0]
                data_block = data_block.astype('double')
                if gender == "m":
                    label = 0
                else:
                    label = 1
                entire_data = data_block,label,fileID
                if fileID in df_train:
                    count_tr += 1
                    train_files.append(entire_data)
                if fileID in df_test:
                    test_files.append(entire_data)
                    count_test+=1
                #print(len(processed_files))
                print(count_tr,count_test)
                
                count_block += 1
    #np.savez(dataset_file,*processed_files)
    np.savez(dataset_file_train,*train_files)
    np.savez(dataset_file_test,*test_files)
               
           
    return dims_array

def get_mfcc_zero_pad(location,dataset_file,timit_files_path):
    smallest = 100
    dims_array = np.array([])
    processed_files = []
    for dirpath, dirnames, filenames in os.walk(location):
        for filename in [f for f in filenames if f.endswith((".npy"))]:
            file_for_mfcc = filename.split("_") # check this see if it still works
            utterance = file_for_mfcc[1].split("-") # gets rid of "-feats"
            wave_file = timit_files_path+str(file_for_mfcc[0])+'/'+str(utterance[0])+'.wav'
            file_data, samplerate = sf.read(wave_file)
            dataf = lr.feature.mfcc(file_data, samplerate,n_mfcc = 13, n_fft=int(0.025*samplerate),hop_length=160)
            second_dim = dataf.shape[1]
            dims_array = np.append(dims_array, second_dim)
            if second_dim > 599:
                dataf = dataf[:,0:599]
            else:
                dims = 599 - second_dim
                pad = np.zeros((dims,dataf.shape[0]))
                array = pad.reshape(dataf.shape[0],dims)
                dataf = np.concatenate((dataf,array),axis=1)
            fileID = filename.split("-")
            fileID = fileID[0]
            gender = filename[0]
            dataf = dataf.astype('double')
            if gender == "m":
                label = 0
            else:
                label = 1
            entire_data = dataf,label,fileID
            processed_files.append(entire_data)
    np.savez(dataset_file,*processed_files)
    

    return dims_array



def get_mfcc_blocks(location,dataset_file,timit_files_path):
    smallest = 1000
    dims_array = np.array([])
    processed_files = []
    for dirpath, dirnames, filenames in os.walk(location):
        for filename in [f for f in filenames if f.endswith((".npy"))]:
            file_for_mfcc = filename.split("_")
            utterance = file_for_mfcc[1].split("-")
            wave_file = timit_files_path+str(file_for_mfcc[0])+'/'+str(utterance[0])+'.wav'
            file_data, samplerate = sf.read(wave_file)
            dataf = lr.feature.mfcc(file_data, samplerate,n_mfcc = 13, n_fft=int(0.025*samplerate),hop_length=160)
            second_dim = dataf.shape[1]
            dims_array = np.append(dims_array, second_dim)
            if second_dim < smallest:
                smallest = second_dim
                print(smallest)
    block_size = 113
    for dirpath, dirnames, filenames in os.walk(location):
        for filename in [f for f in filenames if f.endswith((".npy"))]:
            file_for_mfcc = filename.split("_")
            utterance = file_for_mfcc[1].split("-")
            wave_file = timit_files_path+str(file_for_mfcc[0])+'/'+str(utterance[0])+'.wav'
            file_data, samplerate = sf.read(wave_file)
            dataf = lr.feature.mfcc(file_data, samplerate,n_mfcc = 13, n_fft=int(0.025*samplerate),hop_length=160)
            second_dim = dataf.shape[1]
            dims_array = np.append(dims_array, second_dim)
            block_count = int(second_dim/block_size)
            start = 0
            stop = block_size-1
            count_block = 0
            for i in range(block_count):
                data_block = dataf[:,start:stop]
                start = stop+1
                stop = stop+block_size
                fileID = filename.split("-")
                fileID = fileID[0]+'_'+str(count_block)
                gender = filename[0]
                data_block = data_block.astype('double')
                if gender == "m":
                    label = 0
                else:
                    label = 1
                entire_data = data_block,label,fileID
                processed_files.append(entire_data)
                count_block += 1
    np.savez(dataset_file,*processed_files)
           
       
    return dims_array
    
def get_styrian_sets(location,metadata_file,wavefile_location):
    dims_array = np.array([])
    smallest = 1000
    mfcc_train_block = []
    mfcc_train_zp = []
    mfcc_test_zp = []
    mfcc_test_block = []
    wn_train_zp = []
    wn_train_block = []
    wn_test_zp = []
    wn_test_block = []
    df = pd.read_csv(metadata_file)
    dfm = df.as_matrix()
    class1 = 'East'
    class2 = 'Urb'
    class3 = 'Nor'
    count = 0
    count_train = 0
    count_test = 0
    countone = 0
    counttwo = 0
    countthree = 0
    testone = 0
    testtwo = 0
    testthree = 0
    trainone = 0
    traintwo = 0
    trainthree = 0
    count_zp_test = 0
    count_zp = 0
    for dirpath, dirnames, filenames in os.walk(location): # through where files are
        for filename in [f for f in filenames if f.endswith((".npz"))]:
            data_wn = np.load(location+filename)
            data_wn = np.array(data_wn['pase']) # what to append to wavenet file
            #print(data_wn.shape)
            utterance = filename.split("-") # waveid
            #print(utterance)
            utterance = utterance[0].split('.')
            #data_wn = np.array(data_wn) # what to append to wavenet file
            #utterance = filename.split("-") # waveid
            #print(utterance[0],' utterance')
            wave_file = wavefile_location+'/'+str(utterance[0])+'.wav'
            wave_id =str(utterance[0])+'.wav'
            #print(wave_id,' waveid ')
            #print(wave_id in dfm)
            if wave_id in dfm:
                #print('found')
                count +=1
                a = np.where(dfm[:,3]==wave_id)
                group = dfm[a,0]
                group = str(group)
                label_np = dfm[a,2]
                label_str = str(label_np)
                file_data, samplerate = sf.read(wave_file)
                data_mfcc = lr.feature.mfcc(file_data, samplerate,n_mfcc = 13, n_fft=int(0.025*samplerate),hop_length=160)
                second_dim = data_wn.shape[1]
                data_mfcc = data_mfcc[:,0:second_dim]
                #print(data_mfcc.shape,data_wn.shape)
                dims_array = np.append(dims_array, second_dim)
                if class1 in label_str:
                    label = 0
                    countone +=1
                if class2 in label_str:
                    label = 1
                    counttwo+=1
                if class3 in label_str:
                    label = 2
                    countthree +=1
                #print(label,label_str)
                fileID = str(utterance[0])
                zero_pad_length = 120
                block_size = 42
                
                # is it in test or train group :
                if 'train' in group:
                    count_train +=1
                    start = 0
                    stop = block_size-1
                    count_block = 0
                    if label == 0:
                        trainone+=1
                    if label == 1:
                        traintwo+=1
                    if label == 2:
                        trainthree += 1
                        
                    block_count = int(second_dim/block_size)
                    for i in range(block_count):
                        data_block_wn = data_wn[:,start:stop]
                        data_block_mfcc = data_mfcc[:,start:stop]
                        start = stop+1
                        stop = stop+block_size
                        data_block_mfcc = data_block_mfcc.astype('double')
                        data_block_wn = data_block_wn.astype('double')
                        dbwn = data_block_wn,label,fileID
                        dbmf = data_block_mfcc,label,fileID
                        mfcc_train_block.append(dbmf)
                        wn_train_block.append(dbwn)
                        
                    # now append data for zeropad
                    if second_dim > zero_pad_length:
                        data_wn = data_wn[:,0:zero_pad_length]
                        data_mfcc = data_mfcc[:,0:zero_pad_length]
                    else:
                        dims = zero_pad_length - second_dim
                        pad = np.zeros((dims,data_wn.shape[0]))
                        array = pad.reshape(data_wn.shape[0],dims)
                        data_wn = np.concatenate((data_wn,array),axis=1)
                        
                        dims = zero_pad_length - second_dim
                        pad = np.zeros((dims,data_mfcc.shape[0]))
                        array = pad.reshape(data_mfcc.shape[0],dims)
                        data_mfcc = np.concatenate((data_mfcc,array),axis=1)
                        
                        
                    data_wn = data_wn.astype('double')
                    data_mfcc = data_mfcc.astype('double')
                    wn_data = data_wn,label,fileID
                    mfcc_data = data_mfcc,label,fileID
                    count_zp += 1
                    wn_train_zp.append(wn_data)
                    mfcc_train_zp.append(mfcc_data)
                if 'test' in group:
                    count_test +=1
                    start = 0
                    stop = block_size-1
                    count_block = 0
                    if label == 0:
                        testone+=1
                    if label == 1:
                        testtwo+=1
                    if label == 2:
                        testthree += 1
                    block_count = int(second_dim/block_size)
                    for i in range(block_count):
                        data_block_wn = data_wn[:,start:stop]
                        data_block_mfcc = data_mfcc[:,start:stop]
                        start = stop+1
                        stop = stop+block_size
                        data_block_mfcc = data_block_mfcc.astype('double')
                        data_block_wn = data_block_wn.astype('double')
                        dbwn = data_block_wn,label,fileID
                        dbmf = data_block_mfcc,label,fileID
                        mfcc_test_block.append(dbmf)
                        wn_test_block.append(dbwn)
                        
                    # now append data for zeropad
                    if second_dim > zero_pad_length:
                        data_wn = data_wn[:,0:zero_pad_length]
                        data_mfcc = data_mfcc[:,0:zero_pad_length]
                    else:
                        dims = zero_pad_length - second_dim
                        pad = np.zeros((dims,data_wn.shape[0]))
                        array = pad.reshape(data_wn.shape[0],dims)
                        data_wn = np.concatenate((data_wn,array),axis=1)
                        
                        dims = zero_pad_length - second_dim
                        pad = np.zeros((dims,data_mfcc.shape[0]))
                        array = pad.reshape(data_mfcc.shape[0],dims)
                        data_mfcc = np.concatenate((data_mfcc,array),axis=1)
                    #print('put file in test')
                    count_zp_test += 1
                    data_mfcc = data_mfcc.astype('double')
                    data_wn = data_wn.astype('double')
                    wn_data = data_wn,label,fileID
                    mfcc_data = data_mfcc,label,fileID
                    wn_test_zp.append(wn_data)
                    mfcc_test_zp.append(mfcc_data)
    print(count,count_train,count_test)
    print(countone,counttwo,countthree)
    print(trainone,traintwo,trainthree)
    print(testone,testtwo,testthree)
    print(count_zp,count_zp_test)
                
                
                
                

    #np.savez('/Users/I26259/sty_mfcc_train_zp.npz',*mfcc_train_zp)
    #np.savez('/Users/I26259/sty_mfcc_test_zp.npz',*mfcc_test_zp)
    #np.savez('/Users/I26259/sty_mfcc_train_block.npz',*mfcc_train_block)
    #np.savez('/Users/I26259/sty_mfcc_test_block.npz',*mfcc_test_block)
    np.savez('/Users/I26259/sty_pase_train_block.npz',*wn_train_block)
    np.savez('/Users/I26259/sty_pase_train_zp.npz',*wn_train_zp)
    np.savez('/Users/I26259/sty_pase_test_zp.npz',*wn_test_zp)
    np.savez('/Users/I26259/sty_pase_test_block.npz',*wn_test_block)
           
       
    return dims_array
    



    
def get_music_set_train(location,wavefile_location):
    dims_array = np.array([])
    smallest = 1000
    mfcc_train_block = []
    mfcc_train_zp = []
    wn_train_zp = []
    wn_train_block = []
    count = 0
    for dirpath, dirnames, filenames in os.walk(location): # through where files are
        for filename in [f for f in filenames if f.endswith((".npz"))]:
            data_wn = np.load(location+filename)
            data_wn = np.array(data_wn['pase'])
            #data_wn = np.array(data_wn) # what to append to wavenet file
            utterance = filename.split("_") # waveid
            #print(utterance[0],' utterance')
            print(utterance)
            wave_name = utterance[0]
            class_name = utterance[1].split(".")
            class_name = class_name[0]
            print(wave_name,class_name)
            #utt = utt.split('_')
            #print(utt[0],utt[1])
            wave_file = wavefile_location+'/'+str(class_name)+'/'+str(wave_name)+'.wav'
            wave_id =str(utterance[0])+'.wav'
            #print(wave_id,' waveid ')
            #print(wave_id in dfm)
            file_data, samplerate = sf.read(wave_file)
            #data_mfcc = lr.feature.mfcc(file_data, samplerate,n_mfcc = 13, n_fft=int(0.025*samplerate),hop_length=160)
            second_dim = data_wn.shape[1]
            #data_mfcc = data_mfcc[:,0:second_dim]
            #print(data_mfcc.shape,data_wn.shape)
            dims_array = np.append(dims_array, second_dim)
            label = class_name[5]
            #label = label[5]
            label = int(label)-1
            
            print(label)
            fileID = str(wave_name)
            zero_pad_length = 120
            block_size = 200
            start = 0
            stop = block_size-1
            count_block = 0
                # is it in test or train group :
              
            block_count = int(second_dim/block_size)
            for i in range(block_count):
                data_block_wn = data_wn[:,start:stop]
                #data_block_mfcc = data_mfcc[:,start:stop]
                start = stop+1
                stop = stop+block_size
                #data_block_mfcc = data_block_mfcc.astype('double')
                data_block_wn = data_block_wn.astype('double')
                dbwn = data_block_wn,label,fileID
                #dbmf = data_block_mfcc,label,fileID
                #mfcc_train_block.append(dbmf)
                wn_train_block.append(dbwn)
                        
                    # now append data for zeropad
            if second_dim > zero_pad_length:
                data_wn = data_wn[:,0:zero_pad_length]
                #data_mfcc = data_mfcc[:,0:zero_pad_length]
            else:
                dims = zero_pad_length - second_dim
                pad = np.zeros((dims,data_wn.shape[0]))
                array = pad.reshape(data_wn.shape[0],dims)
                data_wn = np.concatenate((data_wn,array),axis=1)
                        
                dims = zero_pad_length - second_dim
                #pad = np.zeros((dims,data_mfcc.shape[0]))
                #array = pad.reshape(data_mfcc.shape[0],dims)
                #data_mfcc = np.concatenate((data_mfcc,array),axis=1)
                        
                        
            data_wn = data_wn.astype('double')
            #data_mfcc = data_mfcc.astype('double')
            wn_data = data_wn,label,fileID
            #mfcc_data = data_mfcc,label,fileID
            wn_train_zp.append(wn_data)
            #mfcc_train_zp.append(mfcc_data)
                
                
                
                

    #np.savez('/Users/I26259/music_mfcc_train_zp.npz',*mfcc_train_zp)
    #np.savez('/Users/I26259/music_mfcc_test_block.npz',*mfcc_train_block)
    np.savez('/Users/I26259/music_pase_test_block.npz',*wn_train_block)
    np.savez('/Users/I26259/music_pase_test_zp.npz',*wn_train_zp)
    #np.savez('/Users/I26259/sty_wn_train_block.npz',*wn_train_block)
    #np.savez('/Users/I26259/sty_wn_train_zp.npz',*wn_train_zp)
    #np.savez('/Users/I26259/sty_wn_test_zp.npz',*wn_test_zp)
    #np.savez('/Users/I26259/sty_wn_test_block.npz',*wn_test_block)
           
       
    return dims_array
    



def get_orca_sets(location,metadata_file,wavefile_location):
    dims_array = np.array([])
    smallest = 1000
    mfcc_train_block = []
    mfcc_train_zp = []
    mfcc_test_zp = []
    mfcc_test_block = []
    wn_train_zp = []
    wn_train_block = []
    wn_test_zp = []
    wn_test_block = []
    df = pd.read_csv(metadata_file)
    dfm = df.as_matrix()
    class1 = 'noise'
    class2 = 'orca'
    count = 0
    count_train = 0
    count_test = 0
    countone = 0
    counttwo = 0
    testone = 0
    testtwo = 0
    trainone = 0
    traintwo = 0
    count_zp_test = 0
    count_zp = 0
    for dirpath, dirnames, filenames in os.walk(location): # through where files are
        for filename in [f for f in filenames if f.endswith((".npz"))]:
            #print('going thru wave files')
            data_wn = np.load(location+filename)
            #print(type(data_wn))
            #print(location+filename)
            #print(data_wn['pase'])
            data_wn = np.array(data_wn['pase']) # what to append to wavenet file
            #print(data_wn.shape)
            utterance = filename.split("-") # waveid
            #print(utterance)
            utterance = utterance[0].split('.')
            #print(utterance)
            #print(utterance[0],' utterance')
            wave_file = wavefile_location+'/'+str(utterance[0])+'.wav'
            wave_id =str(utterance[0])+'.wav'
            #print(wave_id,' waveid ')
            #print(wave_id in dfm)
            if wave_id in dfm:
                #print('found')
                count +=1
                a = np.where(dfm[:,0]==wave_id)
                label_np = dfm[a,1]
                label_str = str(label_np)
                file_data, samplerate = sf.read(wave_file)
                data_mfcc = lr.feature.mfcc(file_data, samplerate,n_mfcc = 13, n_fft=int(0.025*samplerate),hop_length=160)
                second_dim = data_wn.shape[1]
                data_mfcc = data_mfcc[:,0:second_dim]
                #print(data_mfcc.shape,data_wn.shape)
                dims_array = np.append(dims_array, second_dim)
                if class1 in label_str:
                    label = 0
                    countone +=1
                if class2 in label_str:
                    label = 1
                    counttwo+=1
                #print(label,label_str)
                fileID = str(utterance[0])
                zero_pad_length = 150
                block_size = 50
                
                # is it in test or train group :
                if 'train' in wave_id:
                    count_train +=1
                    start = 0
                    stop = block_size-1
                    count_block = 0
                    if label == 0:
                        trainone+=1
                    if label == 1:
                        traintwo+=1
                        
                    block_count = int(second_dim/block_size)
                    for i in range(block_count):
                        data_block_wn = data_wn[:,start:stop]
                        data_block_mfcc = data_mfcc[:,start:stop]
                        start = stop+1
                        stop = stop+block_size
                        data_block_mfcc = data_block_mfcc.astype('double')
                        data_block_wn = data_block_wn.astype('double')
                        dbwn = data_block_wn,label,fileID
                        dbmf = data_block_mfcc,label,fileID
                        mfcc_train_block.append(dbmf)
                        wn_train_block.append(dbwn)
                        
                    # now append data for zeropad
                    if second_dim > zero_pad_length:
                        data_wn = data_wn[:,0:zero_pad_length]
                        data_mfcc = data_mfcc[:,0:zero_pad_length]
                    else:
                        dims = zero_pad_length - second_dim
                        pad = np.zeros((dims,data_wn.shape[0]))
                        array = pad.reshape(data_wn.shape[0],dims)
                        data_wn = np.concatenate((data_wn,array),axis=1)
                        
                        dims = zero_pad_length - second_dim
                        pad = np.zeros((dims,data_mfcc.shape[0]))
                        array = pad.reshape(data_mfcc.shape[0],dims)
                        data_mfcc = np.concatenate((data_mfcc,array),axis=1)
                        
                        
                    data_wn = data_wn.astype('double')
                    data_mfcc = data_mfcc.astype('double')
                    wn_data = data_wn,label,fileID
                    mfcc_data = data_mfcc,label,fileID
                    count_zp += 1
                    wn_train_zp.append(wn_data)
                    mfcc_train_zp.append(mfcc_data)
                if 'test' in wave_id:
                    count_test +=1
                    start = 0
                    stop = block_size-1
                    count_block = 0
                    if label == 0:
                        testone+=1
                    if label == 1:
                        testtwo+=1
                    block_count = int(second_dim/block_size)
                    for i in range(block_count):
                        data_block_wn = data_wn[:,start:stop]
                        data_block_mfcc = data_mfcc[:,start:stop]
                        start = stop+1
                        stop = stop+block_size
                        data_block_mfcc = data_block_mfcc.astype('double')
                        data_block_wn = data_block_wn.astype('double')
                        dbwn = data_block_wn,label,fileID
                        dbmf = data_block_mfcc,label,fileID
                        mfcc_test_block.append(dbmf)
                        wn_test_block.append(dbwn)
                        
                    # now append data for zeropad
                    if second_dim > zero_pad_length:
                        data_wn = data_wn[:,0:zero_pad_length]
                        data_mfcc = data_mfcc[:,0:zero_pad_length]
                    else:
                        dims = zero_pad_length - second_dim
                        pad = np.zeros((dims,data_wn.shape[0]))
                        array = pad.reshape(data_wn.shape[0],dims)
                        data_wn = np.concatenate((data_wn,array),axis=1)
                        
                        dims = zero_pad_length - second_dim
                        pad = np.zeros((dims,data_mfcc.shape[0]))
                        array = pad.reshape(data_mfcc.shape[0],dims)
                        data_mfcc = np.concatenate((data_mfcc,array),axis=1)
                    #print('put file in test')
                    count_zp_test += 1
                    data_mfcc = data_mfcc.astype('double')
                    data_wn = data_wn.astype('double')
                    wn_data = data_wn,label,fileID
                    mfcc_data = data_mfcc,label,fileID
                    wn_test_zp.append(wn_data)
                    mfcc_test_zp.append(mfcc_data)
    print(count,count_train,count_test)
    print(countone,counttwo)
    print(trainone,traintwo)
    print(testone,testtwo)
    print(count_zp,count_zp_test)
                
                
                
                

    #np.savez('/Users/I26259/orca_mfcc_train_zp.npz',*mfcc_train_zp)
    #np.savez('/Users/I26259/orca_mfcc_test_zp.npz',*mfcc_test_zp)
    #np.savez('/Users/I26259/orca_mfcc_train_block.npz',*mfcc_train_block)
    #np.savez('/Users/I26259/orca_mfcc_test_block.npz',*mfcc_test_block)
    np.savez('/Users/I26259/orca_pase_train_block.npz',*wn_train_block)
    np.savez('/Users/I26259/orca_pase_train_zp.npz',*wn_train_zp)
    np.savez('/Users/I26259/orca_pase_test_zp.npz',*wn_test_zp)
    np.savez('/Users/I26259/orca_pase_test_block.npz',*wn_test_block)
           
       
    return dims_array
    



#dims_array = get_zero_pad("/Users/I26259/Downloads/timit_pase/","/Users/I26259/timit_pase_train_zp.npz","/Users/I26259/timit_pase_test_zp.npz")

dims_array = get_blocks("/Users/I26259/Downloads/timit_pase/","/Users/I26259/timit_pase_train_block.npz","/Users/I26259/timit_pase_test_block.npz")


#dims_array = get_mfcc_blocks("/Users/I26259/timit_feats_balanced_train/","/Users/I26259/mfcc_train_balanced_blocks.npz","/Users/I26259/timit/train/dr2/")

#dims_array = get_mfcc_zero_pad("/Users/I26259/timit_feats_balanced_train/","/Users/I26259/mfcc_train_balanced_zp.npz","/Users/I26259/timit/train/dr2/")
# to look at the historgram of the durations comment out the (np.save .. ) in the function so you can look at it before you save the dataset file
#dims_array = get_styrian_sets('/Users/I26259/Downloads/sty_pase/','/Users/I26259/ComParE2019_styrialectsDS/ComParE2019_styrialectsDS_labels.csv','/Users/I26259/ComParE2019_styrialectsDS/ComParE2019_styrialectsDS/')
#dims_array = get_orca_sets('/Users/I26259/Downloads/orca_pase/','/Users/I26259/Downloads/DeepAL_ComParE/ComParE2019_OrcaActivity/lab/DeepAL_ComParE.csv','/Users/I26259/Downloads/DeepAL_ComParE/ComParE2019_OrcaActivity/wav')
#dims_array = get_music_set_train('/Users/I26259/Downloads/test_in_pase/','/Users/I26259/test_intruments/')
_ = plt.hist(dims_array, bins='auto')  # arguments are passed to np.histogram
plt.title("Histogram with 'auto' bins")

# timit_feats_balanced_test
# timit_feats_balanced_train

plt.show()
