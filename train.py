from argparse import ArgumentParser
import os,pathlib
parser=ArgumentParser(description='select options to train quantum autoencoder')
parser.add_argument('--seed',default=9999,type=int,help='Some number to index the run')
parser.add_argument('--train',default=False,action='store_true',help='train network!')
parser.add_argument('--wires',default=4,type=int,help='number of wires/qubits that the circuit needs to process(AB system)')
parser.add_argument('--trash-qubits',default=1,type=int,help='number of qubits defining the B system, or the reference and trash states!')
parser.add_argument('--shots',default=5000,type=int)
parser.add_argument('--train_n',default=100000,type=int)
parser.add_argument('--valid_n',default=20000,type=int)
parser.add_argument('-b','--batch-size',default=1,type=int)
parser.add_argument('-e','--epochs',default=20,type=int)
parser.add_argument('--backend',default='autograd')
parser.add_argument('--device_name',default='default.qubit',help='device name for the quantum circuit. If you use lightning.kokkos, be sure \
    to set the OMP_PROC_BIND and OMP_n_threads environment variables')
parser.add_argument('--lr',default=0.01,type=float)
parser.add_argument('--save',default=False,action='store_true')
parser.add_argument('--evictable',default=False,action='store_true')
parser.add_argument('--desc',default='Training run')
parser.add_argument('--n_threads',default='8',type=str)
args=parser.parse_args()

if args.device_name=='lightning.kokkos':
    os.environ['OMP_NUM_THREADS']=args.n_threads
    os.environ['OMP_PROC_BIND']='true'
    print(f"Initialized device {args.device_name} with {os.environ['OMP_NUM_THREADS']} threads")
else:
    print(f"Initialized device {args.device_name}")


import glob,time
import helpers.utils as ut
import matplotlib.pyplot as plt
import case_reader as cr
import helpers.path_setter as ps

from loguru import logger
import quantum.architectures as qc
import quantum.losses as loss
import datetime




args.non_trash=args.wires-args.trash_qubits
assert args.non_trash>0,'Need strictly positive dimensional compressed representation of input state!'

train_max_n=args.train_n
valid_max_n=args.valid_n


print(f"args.wires: {args.wires}")
print(f"args.trash_qubits: {args.trash_qubits}")


if args.save:
    save_dir=os.path.join(ps.path_dict['QAE_save'],str(args.seed))
    plot_dir=os.path.join(save_dir,'plots')
    pathlib.Path(plot_dir).mkdir(parents=True,exist_ok=True)
    print("Will save models to: ",save_dir)



logger.add(os.path.join(save_dir,'logs.log'),rotation='10 MB',backtrace=True,diagnose=True,level='DEBUG', mode="w")

# Set device name
device_name=args.device_name
### Initialize the quantum autoencoder ##
qAE=qc.QuantumAutoencoder(wires=args.wires, trash_qubits=args.trash_qubits, dev_name=args.device_name)
qAE.set_circuit(reuploading=True)



cost_fn=loss.batch_semi_classical_cost
qc.print_training_params()

### Initialize the weights randomly ###
init_weights=qc.np.random.uniform(0,qc.np.pi,size=(len(qc.auto_wires)*6,), requires_grad=True)


### Save initial arguments for logging purposes to a text file ###
ut.Pickle(args,'args',path=save_dir)
with open(os.path.join(save_dir,'args.txt'),'w+') as f:
    f.write(repr(args))


### Load the data and create a dataloader ###
train_filelist=sorted(glob.glob(ps.path_dict['QCD_train']+'/*.h5'))
val_filelist=sorted(glob.glob(ps.path_dict['QCD_test']+'/*.h5'))
train_loader = cr.CASEDelphesDataLoader(filelist=train_filelist,batch_size=args.batch_size,input_shape=(len(qc.auto_wires),3),train=True,max_samples=train_max_n)
val_loader = cr.CASEDelphesDataLoader(filelist=val_filelist,batch_size=args.batch_size,input_shape=(len(qc.auto_wires),3),train=False,max_samples=valid_max_n) 


### Initialize the optimizer ###
optimizer=qc.qml.AdamOptimizer(stepsize=args.lr)


### Initialize the trainer ###
trainer=qc.QuantumTrainer(qAE,lr=args.lr,backend_name=args.backend,init_weights=init_weights,device_name=device_name,\
                          train_max_n=train_max_n,valid_max_n=valid_max_n,epochs=args.epochs,batch_size=args.batch_size,\
                            logger=logger,save=args.save,patience=4,optimizer=optimizer,loss_fn=cost_fn)
trainer.print_params('Initialized parameters!')
trainer.set_directories(save_dir)
if args.evictable:
    trainer.is_evictable_job()
### Begin logging ###
logger.info(f"Training started at {datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}")
logger.info(f'Epochs: {args.epochs} | Learning rate: {args.lr} | Batch size: {args.batch_size} \nBackend: {args.backend} | Wires: {args.wires} | Trash qubits: {args.trash_qubits} | Shots: {args.shots} \n')    
logger.info(f'Additional information: {args.desc}')


### Begin training ###
abs_start=time.time()
try:
    history=trainer.run_training_loop(train_loader,val_loader)
except KeyboardInterrupt:
    print("WHYYYYY")
    print("DON'T PRESS CTRL+C AGAIN. I'M TRYING TO SAVE THE CURRENT MODEL AND WRITE TO LOG!") 
    trainer.save(save_dir,name='aborted_weights.pickle')
    trainer.print_params('Training aborted. Current parameters are: ')
finally:
    logger.info('Training completed with the following parameters:')
    trainer.print_params('Trained parameters:')
    history=trainer.fetch_history()
    print (history)
    if args.save:
        done_epochs=len(history['train'])
        ut.Pickle(history,'history',path=save_dir)
        fig,axes=plt.subplots(figsize=(15,12))
        axes.plot(qc.np.arange(done_epochs),history['train'],label='train',linewidth=2)
        axes.plot(qc.np.arange(done_epochs+1),history['val'],label='val',linewidth=2)
        axes.set_xlabel('Epochs',size=25)
        axes.set_ylabel('$1-<T|F> $(in %)',size=25)
        axes.set_xticks(qc.np.arange(0,done_epochs+1,5))
        axes.legend(prop={'size':25})

        axes.tick_params(labelsize=20)
        fig.savefig(os.path.join(save_dir,'history'))
        abs_end=time.time()
        logger.info(f"Training finished at {datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}")
        logger.info(f"Total time taken including all overheads: {abs_end-abs_start:.2f} seconds")
    

