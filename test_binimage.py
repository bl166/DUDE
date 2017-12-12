import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

order_list = [8,9,10,11,12,14,16,18,20]
delta_list = [.01,.02,.05,.10]
size_list = [50,100,150,200]


def corrupt_BSC(input,delta):
    errorbits = np .random.rand(input.shape[0],input.shape[1])<delta
    output = np.array(input);
    output[errorbits]=1-output[errorbits]
    return (output,delta)


def ord2rad(k):
    r = np.int64(np.ceil(np.ceil(np.sqrt(k))/2))
    return r


def padding(input,r):
    padded = np.zeros(tuple([x+r*2 for x in im.shape]))

    padded[r:r+im.shape[0],r:r+im.shape[1]] += input
    padded[:,:r] += padded[:,(2*r):r:-1]
    padded[:,-r:] += padded[:,im.shape[1]:-r][:,::-1]
    padded[:r,:] += padded[(2*r):r:-1,:]
    padded[-r:,:] += padded[im.shape[0]:-r,:][::-1,:]

    return padded


def orderedPath(k):
    path = []
    for r in range(1,ord2rad(k)+1):
        n_chunk = 2*r
        # chunk 1: (-r,0)->(r,0)->(0,-r)->(0,r)
        path += [(-r,0),(r,0),(0,-r),(0,r)]

        for c in range(1,n_chunk-1):
            if c%2: # even
                # chunk 2: (-r,-1)->(-r,1)->(r,-1)->(r,1)
                yy = round((c+1)/2)
                p_temp_1 = [(-r,-yy),(-r,yy),(r,-yy),(r,yy)]
                path += p_temp_1
            else: # odd
                p_temp_2 = [(x[1],x[0]) for x in p_temp_1]
                path += p_temp_2
        # chunk -1: (-r,r)->(-r,r)->(r,-r)->(r,r)
        path += [(-r,r),(-r,r),(r,-r),(r,r)]

    return path[:k]


def dude_bim(im,k,d=None):
    r = ord2rad(k)
    path = orderedPath(k)
    im_flat = im.flatten()
    im_pad = padding(im,r)
    im_denoised = np.zeros(im.shape)
    xx, yy = np.meshgrid(range(r,im.shape[1]+r),range(r,im.shape[0]+r))

    # denoise im_pad[x,y]
    im_pad = padding(im,r)
    neighbors = np.zeros((im.shape[0]*im.shape[1],len(path)))
    count=0
    for x,y in zip(xx.flatten(),yy.flatten()):
        p_temp = [(p[0]+y,p[1]+x) for p in path]
        neighbors[count,:] = [im_pad[p[0],p[1]] for p in p_temp]
        count+=1

    if d is None: # estimate delta
        import fastpy as fp
        d = fp.estimate_loop(np.array(im_flat,dtype=np.int32),np.array(neighbors,dtype=np.int32))
        print("estimated channel param delta = ",d)
    else:
        print("known channel param delta = ",d)

    for x,y in zip(xx.flatten(),yy.flatten()): # denoising starts
        p_temp = [(p[0]+y,p[1]+x) for p in path]
        neighbor_map = (neighbors == [im_pad[p[0],p[1]] for p in p_temp]).all(axis=1)
        # count +im_pad[x,y]
        pos_map = (im == im_pad[y,x]).flatten()
        m1 = (neighbor_map*pos_map).sum()
        # count -im_pad[x,y]
        m0 = (neighbor_map*(~pos_map)).sum()
        # compare to the threshold
        th = 2*d*(1-d)/((1-d)**2+d**2)

        if m0==0:
            im_denoised[y-r,x-r] = im_pad[y,x]
        elif m1/m0>=th:
            im_denoised[y-r,x-r] = im_pad[y,x]
        else:
            im_denoised[y-r,x-r] = 1-im_pad[y,x]
        print("Progress: ",y,x,end="    \r")

    er = np.divide(np.count_nonzero(im!=im_denoised),im.shape[0]*im.shape[1])
    imd_name = './result_figures/corrected_im_size_'+str(SIZE)+'_order_'+str(K)+'_delta_'+str(D)+'_err_'+str(er)+'.png'
    Image.fromarray((im_denoised*255).astype(np.uint8)).save(imd_name)
    return im_denoised



if __name__=="__main__":

    order = []
    delta = []
    error = []
    size = []
    for SIZE in size_list:
        for D in delta_list:
            for K in order_list:

                RAD = ord2rad(K)
                im = np.asarray(Image.open("./source_images/test_"+str(SIZE)+".png")) # bw scan image
                #im = np.asarray(Image.open("lena_halftone.png")) # halftone image
                imc_name = "./corrupted_images/corrupted_im_"+str(SIZE)+"_delta_"+str(D)+".png"
                if os.path.isfile(imc_name):
                    im_corr = np.asarray(Image.open(imc_name))>127
                else:
                    im_corr,_ = corrupt_BSC(im,D)
                    Image.fromarray((im_corr * 255).astype(np.uint8)).save(imc_name)

                im_dude = dude_bim(im_corr,K,D)

            # fig = plt.figure(100)
            # ax1 = fig.add_subplot(131)
            # ax1.imshow(1-im,cmap="Greys")
            # ax1.set_title("Original binary image\n(input to BSC, delta="+str(D)+")")
            # ax2 = fig.add_subplot(132)
            # ax2.imshow(1-im_corr,cmap="Greys")
            # ax2.set_title("Corrupted image\n(output of BSC)\n(input to DUDE)")
            # ax3 = fig.add_subplot(133)
            # ax3.imshow(1-im_dude,cmap="Greys")
            # ax3.set_title("Cleaned image\n(output of DUDE, K="+str(K)+")\nAccuracy = "+str(1-(np.count_nonzero(im_dude-im)/np.size(im))))
            # plt.show()
