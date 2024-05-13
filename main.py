from dataloader import load_data
import pandas as pd
import gc
import os
import torch
import numpy as np
import time
from utils import calc_weight
from utils import evaluate
import random
import anndata as ad
import argparse
import importlib
import scipy
import scipy.io as sio

import warnings
warnings.filterwarnings('ignore')

def main():
    parser = argparse.ArgumentParser(description='Program Description')

    parser.add_argument('--model', type=str, default='moDVTM', 
                    help='''Choose the model to use:
                            moDVTM, moDVTM_nolangda, moDVTM_moe, scMM,
                            moDVTM_rna, moDVTM_dna, moDVTM_protein,
                            moDVTM_share, cobolt, multigrate,
                            moDVTM_e-10, moDVTM_e-9, moDVTM_e-8,
                            moDVTM_e-7, moDVTM_e-6, moDVTM_e-5,
                            moDVTM_e-4, moDVTM_e-3, moDVTM_e-2,
                            moDVTM_e-1, moDVTM_e-0, moDVTM_e+1''')


    parser.add_argument('--task', type=str, default='inte', 
                        help='Choose the task to perform: inte, impu_another2rna, impu_rna2another')

    parser.add_argument('--dataset', type=str, default='nips_multi', 
                        help='Choose the dataset to use: nips_multi, nips_cite, hbic')

    parser.add_argument('--test_ratio', type=float, default=0.1, 
                        help='Specify the ratio of the dataset to be used for testing (e.g., 0.1)')

    parser.add_argument('--num_topic', type=int, default=100, 
                        help='Specify the number of topics')

    parser.add_argument('--num_indep', type=int, default=20, 
                        help='Specify the number of independent components')

    parser.add_argument('--emd_dim', type=int, default=400, 
                        help='Specify the dimension of the embedded space')

    parser.add_argument('--Total_epoch', type=int, default=500, 
                        help='Specify the total number of epochs for training')

    parser.add_argument('--batch_size', type=int, default=2000, 
                        help='Specify the batch size')

    parser.add_argument('--gpu_index', type=str, default='5', 
                        help='Specify the index of the GPU to use')

    parser.add_argument('--seed', type=int, default=3407, 
                        help='Specify the random seed')

    parser.add_argument('--no-return_fig', dest='return_fig', action='store_false',
                        help='Do not return a figure object after evaluation')

    parser.set_defaults(return_fig=True)

    parser.add_argument('--eval_epoch', type=int, default=10, 
                        help='Specify the random seed')

    args = parser.parse_args()

    # 添加评估参数
    parser.add_argument('--batch_col', type=str, 
                        default='batch_indices', help='Column name for batch')
    
    parser.add_argument('--plot_fname', type=str, 
                        default='moDVTM_delta', help='File name for plot')
    
    parser.add_argument('--cell_type_col', type=str, 
                        default='cell_type', help='Column name for cell type')
    
    parser.add_argument('--clustering_method', type=str, 
                        default='louvain', help='Clustering method')
    
    parser.add_argument('--resolutions', nargs='+', 
                        type=float, default=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2], help='Resolutions list')

    args = parser.parse_args()

    model_name = args.model
    task_name = args.task

    if model_name in ['moDVTM_e-10']:
        kl_wei = 1e-10
    elif model_name in ['moDVTM_e-9']:
        kl_wei = 1e-9
    elif model_name in ['moDVTM_e-8']:
        kl_wei = 1e-8
    elif model_name in ['moDVTM_e-7']:
        kl_wei = 1e-7
    elif model_name in ['moDVTM_e-6']:
        kl_wei = 1e-6
    elif model_name in ['moDVTM_e-5']:
        kl_wei = 1e-5
    elif model_name in ['moDVTM_e-4']:
        kl_wei = 1e-4
    elif model_name in ['moDVTM_e-3']:
        kl_wei = 1e-3
    elif model_name in ['moDVTM_e-2']:
        kl_wei = 1e-2
    elif model_name in ['moDVTM_e-1']:
        kl_wei = 1e-1
    elif model_name in ['moDVTM_e-0']:
        kl_wei = 1
    elif model_name in ['moDVTM_e+1']:
        kl_wei = 10
    elif args.task in ['inte']:
        if model_name in ['moDVTM', 'moDVTM_share', 'moDVTM_nolangda', 'moDVTM_moe', 'scMM', 'cobolt', 'multigrate', 'moDVTM_rna', 'moDVTM_dna','moDVTM_protein']:
            kl_wei = 1e-4
    elif args.task in ['impu_another2rna', 'impu_rna2another']:
        if model_name in ['moDVTM', 'moDVTM_share', 'moDVTM_nolangda', 'moDVTM_moe', 'scMM', 'cobolt', 'multigrate', 'moDVTM_rna', 'moDVTM_dna','moDVTM_protein']:
            kl_wei = 1e-7
    # 添加保存文件参数
    result_path = f'./Result_{args.dataset}_{args.task}'
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    if args.task in ['inte']:
        fig = f'./{args.model}_fig'
        result = f'./Result_{args.model}'
        model = f'./Trained_{args.model}'
        plot_dir = os.path.join(result_path, fig)
        model_path = os.path.join(result_path, model)
        output_csv = os.path.join(result_path, result)

        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        if not os.path.exists(model_path):
            os.makedirs(model_path)
        
        if not os.path.exists(output_csv):
            os.makedirs(output_csv)
    
    elif args.task in ['impu_another2rna', 'impu_rna2another']:
        plot_dir = 0
        pth = f'./{args.model}'
        path = os.path.join(result_path, pth)
        if not os.path.exists(path):
            os.makedirs(path)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_index
    eval_epoch = args.eval_epoch

    # Evaluation parameters
    # 构建 Eval_kwargs 字典
    Eval_kwargs = {
        'batch_col': args.batch_col,
        'plot_fname': args.plot_fname,
        'cell_type_col': args.cell_type_col,
        'clustering_method': args.clustering_method,
        'resolutions': args.resolutions,
        'plot_dir': plot_dir
    }

    X_mod1_train_T, X_mod2_train_T, X_mod1_test_T, X_mod2_test_T, batch_index_train_T, batch_index_test_T, mod1_test,  mod1_train, test_mod1_sum, test_mod2_sum= load_data(args.dataset, args.test_ratio, args.seed)
    # if args.task in ['impu_another2rna', 'impu_rna2another']:
    #     csv_path = os.path.join(path, './nips_obs.csv')
    #     X_mod1_test_T.obs.to_csv(csv_path)

    
    num_batch = len(batch_index_train_T.unique())
    input_dim_mod1 = X_mod1_train_T.shape[1]
    input_dim_mod2 = X_mod2_train_T.shape[1]
    train_num = X_mod1_train_T.shape[0]

    module = importlib.import_module('model')

    if model_name in ['moDVTM', 'moDVTM_share', 'moDVTM_nolangda', 'moDVTM_moe', 'scMM', 'cobolt', 'multigrate']:
        function_name = f'build_{model_name}'
        build_model_function = getattr(module, function_name)
        trainer_name = f'Trainer_{model_name}_{task_name}'
        trainer_model_function = getattr(module, trainer_name)
        encoder_mod1, encoder_mod2, decoder, optimizer = build_model_function(input_dim_mod1, input_dim_mod2, num_batch, num_topic=args.num_topic, num_indep = args.num_indep, emd_dim=args.emd_dim)
        trainer = trainer_model_function(encoder_mod1, encoder_mod2, decoder, optimizer)
    elif model_name in ['moDVTM_e-10', 'moDVTM_e-9', 'moDVTM_e-8', 'moDVTM_e-7', 
                      'moDVTM_e-6', 'moDVTM_e-5', 'moDVTM_e-4', 'moDVTM_e-3', 'moDVTM_e-2', 'moDVTM_e-1', 'moDVTM_e-0', 'moDVTM_e+1']:
        function_name = f'build_moDVTM'
        build_model_function = getattr(module, function_name)
        trainer_name = f'Trainer_moDVTM_{task_name}'
        trainer_model_function = getattr(module, trainer_name)
        encoder_mod1, encoder_mod2, decoder, optimizer = build_model_function(input_dim_mod1, input_dim_mod2, num_batch, num_topic=args.num_topic, num_indep = args.num_indep, emd_dim=args.emd_dim)
        trainer = trainer_model_function(encoder_mod1, encoder_mod2, decoder, optimizer)
    elif model_name in ['moDVTM_rna']:
        function_name = f'build_{model_name}'
        build_model_function = getattr(module, function_name)
        trainer_name = f'Trainer_{model_name}_{task_name}'
        trainer_model_function = getattr(module, trainer_name)
        encoder_mod1, decoder, optimizer = build_model_function(input_dim_mod1, num_batch, num_topic=args.num_topic, num_indep = args.num_indep, emd_dim=args.emd_dim)
        trainer = trainer_model_function(encoder_mod1, decoder, optimizer)
    elif model_name in ['moDVTM_dna','moDVTM_protein']:
        function_name = f'build_{model_name}'
        build_model_function = getattr(module, function_name)
        trainer_name = f'Trainer_{model_name}_{task_name}'
        trainer_model_function = getattr(module, trainer_name)
        encoder_mod1, decoder, optimizer = build_model_function(input_dim_mod2, num_batch, num_topic=args.num_topic, num_indep = args.num_indep, emd_dim=args.emd_dim)
        trainer = trainer_model_function(encoder_mod1, decoder, optimizer)

        
    LIST = list(np.arange(0, train_num))
    
    if args.task in ['inte']:
        EPOCH = []
        ARI = []
        NMI = []
        ASW = []
        ASW_2 = []
        B_kBET = []
        B_ASW = []
        B_GS = []
        B_ebm = []

        best_ari = 0

        for epoch in range(args.Total_epoch):
            Loss_all = 0
            NLL_all_mod1 = 0
            NLL_all_mod2 = 0
            KL_all = 0

            tstart = time.time()

            np.random.shuffle(LIST)
            KL_weight = calc_weight(epoch, args.Total_epoch, 0, 1 / 3, 0, kl_wei)
            if model_name in ['moDVTM', 'moDVTM_share', 'moDVTM_nolangda', 'moDVTM_moe', 'scMM', 'cobolt', 'multigrate', 'moDVTM_e-10', 'moDVTM_e-9', 'moDVTM_e-8', 'moDVTM_e-7', 
                              'moDVTM_e-6', 'moDVTM_e-5', 'moDVTM_e-4', 'moDVTM_e-3', 'moDVTM_e-2', 'moDVTM_e-1', 'moDVTM_e-0', 'moDVTM_e+1']:
                for iteration in range(train_num // args.batch_size):
                    x_minibatch_mod1_T = X_mod1_train_T[LIST[iteration * args.batch_size: (iteration + 1) * args.batch_size], :].to('cuda')
                    x_minibatch_mod2_T = X_mod2_train_T[LIST[iteration * args.batch_size: (iteration + 1) * args.batch_size], :].to('cuda')
                    batch_minibatch_T = batch_index_train_T[LIST[iteration * args.batch_size: (iteration + 1) * args.batch_size]].to('cuda')

                    loss, nll_mod1, nll_mod2, kl = trainer.train(x_minibatch_mod1_T, x_minibatch_mod2_T, batch_minibatch_T, KL_weight)

                    Loss_all += loss
                    NLL_all_mod1 += nll_mod1
                    NLL_all_mod2 += nll_mod2
                    KL_all += kl
                    

                if (epoch % eval_epoch == 0):

                    trainer.encoder_mod1.to('cpu')
                    trainer.encoder_mod2.to('cpu')

                    embed = trainer.get_embed(X_mod1_test_T, X_mod2_test_T)
                    mod1_test.obsm.update(embed)
                    Result = evaluate(adata=mod1_test, n_epoch=epoch, return_fig=args.return_fig, **Eval_kwargs)
                    tend = time.time()
                    print('epoch=%d, Time=%.4f, Cell_ARI=%.4f, Cell_NMI=%.4f, Cell_ASW=%.4f, Cell_ASW2=%.4f, Batch_KBET=%.4f, Batch_ASW=%.4f, Batch_GC=%.4f, Batch_ebm=%.4f' % (
                        epoch, tend-tstart, Result['ari'], Result['nmi'], Result['asw'], Result['asw_2'], Result['k_bet'], Result['batch_asw'], Result['batch_graph_score'], Result['ebm']))

                    trainer.encoder_mod1.cuda()
                    trainer.encoder_mod2.cuda()

                    EPOCH.append(epoch)
                    ARI.append(Result['ari'])
                    NMI.append(Result['nmi'])
                    ASW.append(Result['asw'])
                    ASW_2.append(Result['asw_2'])
                    B_kBET.append(Result['k_bet'])
                    B_ASW.append(Result['batch_asw'])
                    B_GS.append(Result['batch_graph_score'])
                    B_ebm.append(Result['ebm'])

                    df = pd.DataFrame.from_dict(
                        {
                            'Epoch': pd.Series(EPOCH),
                            'ARI': pd.Series(ARI),
                            'NMI': pd.Series(NMI),
                            'ASW': pd.Series(ASW),
                            'ASW_2': pd.Series(ASW_2),
                            'B_kBET': pd.Series(B_kBET),
                            'B_ASW': pd.Series(B_ASW),
                            'B_GC': pd.Series(B_GS),
                            'B_ebm': pd.Series(B_ebm),
                        }
                    )
                    csv_pth = os.path.join(output_csv, 'data.csv')
                    df.to_csv(csv_pth)

                    # 构建模型保存路径
                    encoder1_path = os.path.join(model_path, 'model_encoder1.pth')
                    encoder2_path = os.path.join(model_path, 'model_encoder2.pth')
                    decoder_path = os.path.join(model_path, 'model_decoder.pth')

                    if Result['ari']>best_ari:
                        best_ari = Result['ari']
                        torch.save(trainer.encoder_mod1.state_dict(), encoder1_path)
                        torch.save(trainer.encoder_mod2.state_dict(), encoder2_path)
                        torch.save(trainer.decoder.state_dict(), decoder_path)
            elif model_name in ['moDVTM_rna']:
                for iteration in range(train_num // args.batch_size):
                    x_minibatch_mod1_T = X_mod1_train_T[LIST[iteration * args.batch_size: (iteration + 1) * args.batch_size], :].to('cuda')
                    #x_minibatch_mod2_T = X_mod2_train_T[LIST[iteration * args.batch_size: (iteration + 1) * args.batch_size], :].to('cuda')
                    batch_minibatch_T = batch_index_train_T[LIST[iteration * args.batch_size: (iteration + 1) * args.batch_size]].to('cuda')

                    loss, nll_mod1, kl = trainer.train(x_minibatch_mod1_T, batch_minibatch_T, KL_weight)

                    Loss_all += loss
                    NLL_all_mod1 += nll_mod1
                    #NLL_all_mod2 += nll_mod2
                    KL_all += kl
                    

                if (epoch % eval_epoch == 0):

                    trainer.encoder_mod1.to('cpu')
                    #trainer.encoder_mod2.to('cpu')

                    embed = trainer.get_embed(X_mod1_test_T)
                    mod1_test.obsm.update(embed)
                    Result = evaluate(adata=mod1_test, n_epoch=epoch, return_fig=args.return_fig, **Eval_kwargs)
                    tend = time.time()
                    print('epoch=%d, Time=%.4f, Cell_ARI=%.4f, Cell_NMI=%.4f, Cell_ASW=%.4f, Cell_ASW2=%.4f, Batch_KBET=%.4f, Batch_ASW=%.4f, Batch_GC=%.4f, Batch_ebm=%.4f' % (
                        epoch, tend-tstart, Result['ari'], Result['nmi'], Result['asw'], Result['asw_2'], Result['k_bet'], Result['batch_asw'], Result['batch_graph_score'], Result['ebm']))

                    trainer.encoder_mod1.cuda()
                    #trainer.encoder_mod2.cuda()

                    EPOCH.append(epoch)
                    ARI.append(Result['ari'])
                    NMI.append(Result['nmi'])
                    ASW.append(Result['asw'])
                    ASW_2.append(Result['asw_2'])
                    B_kBET.append(Result['k_bet'])
                    B_ASW.append(Result['batch_asw'])
                    B_GS.append(Result['batch_graph_score'])
                    B_ebm.append(Result['ebm'])

                    df = pd.DataFrame.from_dict(
                        {
                            'Epoch': pd.Series(EPOCH),
                            'ARI': pd.Series(ARI),
                            'NMI': pd.Series(NMI),
                            'ASW': pd.Series(ASW),
                            'ASW_2': pd.Series(ASW_2),
                            'B_kBET': pd.Series(B_kBET),
                            'B_ASW': pd.Series(B_ASW),
                            'B_GC': pd.Series(B_GS),
                            'B_ebm': pd.Series(B_ebm),
                        }
                    )

                    csv_pth = os.path.join(output_csv, 'data.csv')
                    df.to_csv(csv_pth)

                    # 构建模型保存路径
                    encoder1_path = os.path.join(model_path, 'model_encoder1.pth')
                    #encoder2_path = os.path.join(model_path, 'model_encoder2.pth')
                    decoder_path = os.path.join(model_path, 'model_decoder.pth')

                    if Result['ari']>best_ari:
                        best_ari = Result['ari']
                        torch.save(trainer.encoder_mod1.state_dict(), encoder1_path)
                        #torch.save(trainer.encoder_mod2.state_dict(), encoder2_path)
                        torch.save(trainer.decoder.state_dict(), decoder_path)
            elif model_name in ['moDVTM_dna','moDVTM_protein']:
                for iteration in range(train_num // args.batch_size):
                    #x_minibatch_mod1_T = X_mod1_train_T[LIST[iteration * args.batch_size: (iteration + 1) * args.batch_size], :].to('cuda')
                    x_minibatch_mod2_T = X_mod2_train_T[LIST[iteration * args.batch_size: (iteration + 1) * args.batch_size], :].to('cuda')
                    batch_minibatch_T = batch_index_train_T[LIST[iteration * args.batch_size: (iteration + 1) * args.batch_size]].to('cuda')

                    loss, nll_mod2, kl = trainer.train( x_minibatch_mod2_T, batch_minibatch_T, KL_weight)

                    Loss_all += loss
                    #NLL_all_mod1 += nll_mod1
                    NLL_all_mod2 += nll_mod2
                    KL_all += kl
                    

                if (epoch % eval_epoch == 0):

                    trainer.encoder_mod1.to('cpu')
                    #trainer.encoder_mod2.to('cpu')

                    embed = trainer.get_embed(X_mod2_test_T)
                    mod1_test.obsm.update(embed)
                    Result = evaluate(adata=mod1_test, n_epoch=epoch, return_fig=args.return_fig, **Eval_kwargs)
                    tend = time.time()
                    print('epoch=%d, Time=%.4f, Cell_ARI=%.4f, Cell_NMI=%.4f, Cell_ASW=%.4f, Cell_ASW2=%.4f, Batch_KBET=%.4f, Batch_ASW=%.4f, Batch_GC=%.4f, Batch_ebm=%.4f' % (
                        epoch, tend-tstart, Result['ari'], Result['nmi'], Result['asw'], Result['asw_2'], Result['k_bet'], Result['batch_asw'], Result['batch_graph_score'], Result['ebm']))

                    trainer.encoder_mod1.cuda()
                    #trainer.encoder_mod2.cuda()

                    EPOCH.append(epoch)
                    ARI.append(Result['ari'])
                    NMI.append(Result['nmi'])
                    ASW.append(Result['asw'])
                    ASW_2.append(Result['asw_2'])
                    B_kBET.append(Result['k_bet'])
                    B_ASW.append(Result['batch_asw'])
                    B_GS.append(Result['batch_graph_score'])
                    B_ebm.append(Result['ebm'])

                    df = pd.DataFrame.from_dict(
                        {
                            'Epoch': pd.Series(EPOCH),
                            'ARI': pd.Series(ARI),
                            'NMI': pd.Series(NMI),
                            'ASW': pd.Series(ASW),
                            'ASW_2': pd.Series(ASW_2),
                            'B_kBET': pd.Series(B_kBET),
                            'B_ASW': pd.Series(B_ASW),
                            'B_GC': pd.Series(B_GS),
                            'B_ebm': pd.Series(B_ebm),
                        }
                    )

                    csv_pth = os.path.join(output_csv, 'data.csv')
                    df.to_csv(csv_pth)

                    # 构建模型保存路径
                    encoder1_path = os.path.join(model_path, 'model_encoder1.pth')
                    #encoder2_path = os.path.join(model_path, 'model_encoder2.pth')
                    decoder_path = os.path.join(model_path, 'model_decoder.pth')

                    if Result['ari']>best_ari:
                        best_ari = Result['ari']
                        torch.save(trainer.encoder_mod1.state_dict(), encoder1_path)
                        #torch.save(trainer.encoder_mod2.state_dict(), encoder2_path)
                        torch.save(trainer.decoder.state_dict(), decoder_path)
    if args.task in ['impu_another2rna', 'impu_rna2another']:
        X_mod1, X_mod2, batch_index = X_mod1_train_T, X_mod2_train_T, batch_index_train_T
        test_X_mod1, test_X_mod2, batch_index_test = X_mod1_test_T, X_mod2_test_T, batch_index_test_T

        for epoch in range(args.Total_epoch):
            Loss_all = 0
            NLL_all_mod1 = 0
            NLL_all_mod2 = 0
            KL_all = 0

            tstart = time.time()

            np.random.shuffle(LIST)
            KL_weight = kl_wei

            for iteration in range(train_num // args.batch_size):
                x_minibatch_mod1_T = X_mod1[LIST[iteration * args.batch_size: (iteration + 1) * args.batch_size], :].to('cuda')
                x_minibatch_mod2_T = X_mod2[LIST[iteration * args.batch_size: (iteration + 1) * args.batch_size], :].to('cuda')
                batch_minibatch_T = batch_index[LIST[iteration * args.batch_size: (iteration + 1) * args.batch_size]]

                loss, nll_mod1, nll_mod2, kl = trainer.train(x_minibatch_mod1_T, x_minibatch_mod2_T, batch_minibatch_T, KL_weight)

                Loss_all += loss
                NLL_all_mod1 += nll_mod1
                NLL_all_mod2 += nll_mod2
                KL_all += kl

            if epoch % eval_epoch == 0:

                trainer.encoder_mod1.to('cpu')
                trainer.encoder_mod2.to('cpu')
                trainer.decoder.to('cpu')

                recon_mod1, recon_mod2 = trainer.reconstruction(test_X_mod1, test_X_mod2, batch_index_test)
                tend = time.time()

                if args.task in ['impu_rna2another']:
                    recon_mod = np.array(recon_mod2) * test_mod2_sum[:, np.newaxis]
                    gt_mod = np.array(test_X_mod2) * test_mod2_sum[:, np.newaxis]


                elif args.task in ['impu_another2rna']:
                    recon_mod = np.array(recon_mod1) * test_mod1_sum[:, np.newaxis]
                    gt_mod = np.array(test_X_mod1) * test_mod1_sum[:, np.newaxis]

                else:
                    print('Wrong Direction!')

                ### save impute results
                if (epoch%100==90):
                    recon_path1 = os.path.join(path, './recon_mod_epoch'+str(epoch)+'.npy')
                    recon_path2 = os.path.join(path, './recon_mod_epoch'+str(epoch)+'.mat')
                    recon_path3 = os.path.join(path, './gt_mod_epoch'+str(epoch)+'.npy')
                    recon_path4 = os.path.join(path, './gt_mod_epoch'+str(epoch)+'.mat')
                    np.save(recon_path1 ,recon_mod)
                    sio.savemat(recon_path2, {'recon':recon_mod})
                    np.save(recon_path3, gt_mod)
                    sio.savemat(recon_path4, {'gt':gt_mod})


                recon_mod_tmp = np.squeeze(recon_mod.reshape([1, -1]))
                gt_mod_tmp = np.squeeze(gt_mod.reshape([1, -1]))


                recon_mod_tmp = np.log(1+recon_mod_tmp)
                gt_mod_tmp = np.log(1+gt_mod_tmp)
                Pearson = scipy.stats.pearsonr(recon_mod_tmp, gt_mod_tmp)[0]
                Spearmanr = scipy.stats.spearmanr(recon_mod_tmp, gt_mod_tmp)[0]

                print('[epoch %0d finished time %4f], Pearson_1=%.4f, Spearmanr_1=%.4f' % (epoch, tend - tstart, Pearson, Spearmanr))

                trainer.encoder_mod1.cuda()
                trainer.encoder_mod2.cuda()
                trainer.decoder.cuda()
                


if __name__ == "__main__":
    main()
