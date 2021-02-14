from __future__ import print_function, absolute_import
import numpy as np
import shutil
import os
import os.path as osp
import cv2
import math

from .tools import mkdir_if_missing

from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics

__all__ = ['visualize_ranked_results']

GRID_SPACING = 10
QUERY_EXTRA_SPACING = 90
BW = 5 # border width
GREEN = (0, 255, 0)
RED = (0, 0, 255)


def visualize_ranked_results(
    distmat, dataset, data_type, width=128, height=256, save_dir='', topk=10
):
    """Visualizes ranked results.

    Supports both image-reid and video-reid.

    For image-reid, ranks will be plotted in a single figure. For video-reid, ranks will be
    saved in folders each containing a tracklet.

    Args:
        distmat (numpy.ndarray): distance matrix of shape (num_query, num_gallery).
        dataset (tuple): a 2-tuple containing (query, gallery), each of which contains
            tuples of (img_path(s), pid, camid, dsetid).
        data_type (str): "image" or "video".
        width (int, optional): resized image width. Default is 128.
        height (int, optional): resized image height. Default is 256.
        save_dir (str): directory to save output images.
        topk (int, optional): denoting top-k images in the rank list to be visualized.
            Default is 10.
    """
    num_q, num_g = distmat.shape
    
    np.set_printoptions(suppress=True)
    
    model = AgglomerativeClustering(affinity='precomputed', n_clusters=750, linkage='complete').fit(distmat)
    #model = AgglomerativeClustering(affinity='precomputed', n_clusters=None, linkage='complete', distance_threshold=350).fit(distmat)
    labels = model.labels_
    print(labels)
    indices = []
    
    '''
    for i,x in enumerate(labels):
        if(x==labels[7]):
            print(dataset[0][i][0])
    '''        
            
    for i,x in enumerate(labels):
        indices.append([])
        for i2,x2 in enumerate(labels):
            if(x!=-1 and x==x2 and i2!=0): 
                indices[i].append(i2)
                labels[i2]=-1
         
    
    indices[0].append(0)
    indices = list(filter(None, indices))
    #print(indices)
    
    '''
    for x in indices[7]:
        print(dataset[0][x][0])
    '''
    
    
    dst = '/notebooks/log/resnet50/output'
    
    for n,i in enumerate(indices):
        dstt = osp.join(dst, str(n)+'x') #dataset[0][i[0]][0].split('/')[-1][0:4])
        mkdir_if_missing(dstt)
        for j in i:
            shutil.copy(dataset[0][j][0], dstt)
    
    for i in os.listdir('/notebooks/log/resnet50/output'):
        listt = list(map(lambda x:x[0:4], os.listdir(os.path.join('/notebooks/log/resnet50/output',i))))
        dict = {j : listt.count(j) for j in listt}
        name = sorted(dict.items(), key = lambda x:x[1])[-1][0]
        if(osp.exists(os.path.join('/notebooks/log/resnet50/output',name+'_4'))):
            os.rename(os.path.join('/notebooks/log/resnet50/output',i),os.path.join('/notebooks/log/resnet50/output',name+'_5'))
        elif(osp.exists(os.path.join('/notebooks/log/resnet50/output',name+'_3'))):
            os.rename(os.path.join('/notebooks/log/resnet50/output',i),os.path.join('/notebooks/log/resnet50/output',name+'_4'))
        elif(osp.exists(os.path.join('/notebooks/log/resnet50/output',name+'_2'))):
            os.rename(os.path.join('/notebooks/log/resnet50/output',i),os.path.join('/notebooks/log/resnet50/output',name+'_3'))
        elif(osp.exists(os.path.join('/notebooks/log/resnet50/output',name))):
            os.rename(os.path.join('/notebooks/log/resnet50/output',i),os.path.join('/notebooks/log/resnet50/output',name+'_2'))      
        else:
            os.rename(os.path.join('/notebooks/log/resnet50/output',i),os.path.join('/notebooks/log/resnet50/output',name))    
    
    
    print("silhouette_score: "+str(metrics.silhouette_score(distmat, labels, metric='euclidean')))
    print("calinski_harabasz_score: "+str(metrics.calinski_harabasz_score(distmat, labels)))
    print("davies_bouldin_score: "+str(metrics.davies_bouldin_score(distmat, labels)))
    
    
    print("Calculating custom metric")
    TP = 0
    FP = 0
    FN = 0
    for i in os.listdir('/notebooks/log/resnet50/output'):
        listt = list(map(lambda x:x[0:4], os.listdir(os.path.join('/notebooks/log/resnet50/output',i))))
        correct = listt.count(i[0:4])
        TP += correct
        FP += len(listt)-correct
        queryList = list(map(lambda x:x[0:4], os.listdir('/notebooks/reid-data/market1501/Market-1501-v15.09.15/query')))
        FN += queryList.count(i)-correct
    
    FMI = TP / math.sqrt((TP + FP) * (TP + FN))
    print("TP: "+str(TP))
    print("FP: "+str(FP))
    print("FN: "+str(FN))
    print("Fowlkes-Mallows Score: "+str(FMI))
    print('Stalling')
    while True:
        pass
    mkdir_if_missing(save_dir)

    print('# query: {}\n# gallery {}'.format(num_q, num_g))
    print('Visualizing top-{} ranks ...'.format(topk))

    query, gallery = dataset
    assert num_q == len(query)
    assert num_g == len(gallery)

    indices = np.argsort(distmat, axis=1)

    def _cp_img_to(src, dst, rank, prefix, matched=False):
        """
        Args:
            src: image path or tuple (for vidreid)
            dst: target directory
            rank: int, denoting ranked position, starting from 1
            prefix: string
            matched: bool
        """
        if isinstance(src, (tuple, list)):
            if prefix == 'gallery':
                suffix = 'TRUE' if matched else 'FALSE'
                dst = osp.join(
                    dst, prefix + '_top' + str(rank).zfill(3)
                ) + '_' + suffix
            else:
                dst = osp.join(dst, prefix + '_top' + str(rank).zfill(3))
            mkdir_if_missing(dst)
            for img_path in src:
                shutil.copy(img_path, dst)
        else:
            dst = osp.join(
                dst, prefix + '_top' + str(rank).zfill(3) + '_name_' +
                osp.basename(src)
            )
            shutil.copy(src, dst)

    for q_idx in range(num_q):
        qimg_path, qpid, qcamid = query[q_idx][:3]
        qimg_path_name = qimg_path[0] if isinstance(
            qimg_path, (tuple, list)
        ) else qimg_path

        if data_type == 'image':
            qimg = cv2.imread(qimg_path)
            qimg = cv2.resize(qimg, (width, height))
            qimg = cv2.copyMakeBorder(
                qimg, BW, BW, BW, BW, cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )
            # resize twice to ensure that the border width is consistent across images
            qimg = cv2.resize(qimg, (width, height))
            num_cols = topk + 1
            grid_img = 255 * np.ones(
                (
                    height,
                    num_cols*width + topk*GRID_SPACING + QUERY_EXTRA_SPACING, 3
                ),
                dtype=np.uint8
            )
            grid_img[:, :width, :] = qimg
        else:
            qdir = osp.join(
                save_dir, osp.basename(osp.splitext(qimg_path_name)[0])
            )
            mkdir_if_missing(qdir)
            _cp_img_to(qimg_path, qdir, rank=0, prefix='query')

        rank_idx = 1
        for g_idx in indices[q_idx, :]:
            gimg_path, gpid, gcamid = gallery[g_idx][:3]
            invalid = (qpid == gpid) & (qcamid == gcamid)

            if not invalid:
                matched = gpid == qpid
                if data_type == 'image':
                    border_color = GREEN if matched else RED
                    gimg = cv2.imread(gimg_path)
                    gimg = cv2.resize(gimg, (width, height))
                    gimg = cv2.copyMakeBorder(
                        gimg,
                        BW,
                        BW,
                        BW,
                        BW,
                        cv2.BORDER_CONSTANT,
                        value=border_color
                    )
                    gimg = cv2.resize(gimg, (width, height))
                    start = rank_idx*width + rank_idx*GRID_SPACING + QUERY_EXTRA_SPACING
                    end = (
                        rank_idx+1
                    ) * width + rank_idx*GRID_SPACING + QUERY_EXTRA_SPACING
                    grid_img[:, start:end, :] = gimg
                else:
                    _cp_img_to(
                        gimg_path,
                        qdir,
                        rank=rank_idx,
                        prefix='gallery',
                        matched=matched
                    )

                rank_idx += 1
                if rank_idx > topk:
                    break

        if data_type == 'image':
            imname = osp.basename(osp.splitext(qimg_path_name)[0])
            cv2.imwrite(osp.join(save_dir, imname + '.jpg'), grid_img)

        if (q_idx+1) % 100 == 0:
            print('- done {}/{}'.format(q_idx + 1, num_q))

    print('Done. Images have been saved to "{}" ...'.format(save_dir))
