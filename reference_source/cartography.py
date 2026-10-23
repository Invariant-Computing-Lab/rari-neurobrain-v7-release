"""Declared ontology, with proxy labels and explicit unmeasured layers."""
import numpy as np

LAYERS = ["cytoarchitectonic", "cell_population", "morphological", "structural_connectome",
          "tract", "synaptic", "functional_connectivity", "effective_connectivity", "spike",
          "spike_train", "spectral", "temporal_dynamics", "latent_manifold", "state_transition",
          "attractor_basin", "plasticity", "receptor", "transcriptomic", "metabolic_resource",
          "lesion", "stimulation_causal", "sensorimotor", "vMEA"]


def summarize(trajectory):
    # trajectory [ticks, parcels, two electrodes + spike_count + action].
    x=np.asarray(trajectory,float)
    if x.ndim!=3 or x.shape[1:]!=(101,4) or not np.isfinite(x).all(): raise ValueError("CARTOGRAPHY_SHAPE")
    electrode=x[:,:,0]
    centered=electrode-electrode.mean(0)
    norms=np.sqrt((centered**2).sum(0));den=np.outer(norms,norms)
    fc=np.divide(centered.T@centered,den,out=np.zeros((101,101)),where=den>1e-12)
    spectrum=(np.abs(np.fft.rfft(centered,axis=0))**2).mean(axis=1)
    layers={name:{"status":"NOT_MEASURED"} for name in LAYERS}
    layers.update({"functional_connectivity":{"status":"SIMULATED_OBSERVATION","matrix":fc.tolist(),"zero_variance_channels":int((norms<1e-12).sum())},
                   "spectral":{"status":"SIMULATED_OBSERVATION","mean_power":spectrum.tolist(),"sampling_unit":"simulation tick"},
                   "spike_train":{"status":"SIMULATED_OBSERVATION","counts":x[:,:,2].tolist()},
                   "temporal_dynamics":{"status":"SIMULATED_OBSERVATION","ticks":len(x)},
                   "sensorimotor":{"status":"SIMULATED_OBSERVATION","mean_action":x[:,:,3].mean(axis=1).tolist()},
                   "vMEA":{"status":"DECLARED_LIMITED_INTERFACE","electrodes_per_parcel":2},
                   "effective_connectivity":{"status":"NOT_ESTIMATED","reason":"correlation is not causal influence"}})
    for name in ("morphological","receptor","transcriptomic","metabolic_resource"):
        layers[name]={"status":"PROXY_SCHEMA_ONLY","biological_measurement":False}
    return {"ontology":"22 source layers plus explicit vMEA", "layers":layers,
            "status":"ENGINEERING_FIXTURE_ONLY"}



def measured(record):
    """Full computational map; unavailable biological layers remain proxies."""
    from .subject import Brain
    from .mapping import atlas,public_atlas
    x=np.asarray(record['trajectory']);result=summarize(x);layers=result['layers']
    states=record['checkpoint'];initial=Brain().checkpoint()
    adjacency=[[i+1,((i+1)%101)+1] for i in range(101)]
    layers['structural_connectome']={'status':'SIMULATED_DEFINED','parcel_ring_edges':adjacency,
                                    'local_edges':[[s['parcel'],j,(j+1)%len(s['weights'])] for s in states for j in range(len(s['weights']))]}
    layers['tract']={'status':'SIMULATED_DEFINED','delay_ticks':8,'weights':.1,'edges':adjacency}
    layers['cell_population']={'status':'SIMULATED_STATE','N4':[row for s in states for row in s['n4']],
                                'npus':806,'N4_coordinates':['r','i','p','o']}
    layers['cytoarchitectonic']={'status':'COMPUTATIONAL_ATLAS_PROXY','cortex_parcels':68,'deep_gray_limbic':16,'cerebellar':12,'brainstem':5}
    layers['synaptic']={'status':'SIMULATED_STATE','weights':[s['weights'] for s in states],'local_delay_ticks':1}
    layers['plasticity']={'status':'SIMULATED_MEASURED','weight_changes':[(np.asarray(s['weights'])-b['weights']).tolist() for s,b in zip(states,initial)]}
    causal=np.asarray(record['causal_paired_difference'])
    layers['effective_connectivity']={'status':'PAIRED_INTERVENTION_MEASURED','target_parcel':record['perturbations'][8]['parcel'],
        'perturbation':record['perturbations'][8]['kind'],'mean_channel_effect':causal.mean(0).tolist(),
        'same_inputs_same_checkpoint':True,'unmeasured_edges':'NOT_ESTIMATED','causal_scope':'frozen eight-tick block input; not total closed-loop effect'}
    layers['stimulation_causal']={'status':'SIMULATED_MEASURED','paired_difference':causal.tolist(),'interventions':record['perturbations']}
    layers['lesion']={'status':'SIMULATED_MEASURED' if record['perturbations'][8]['kind']=='parcel_lesion' else 'NOT_APPLIED_THIS_EPISODE',
                     'target':record['perturbations'][8]['parcel']}
    flat=x.reshape(len(x),-1);mu=flat.mean(0);u,s,v=np.linalg.svd(flat-mu,full_matrices=False)
    layers['latent_manifold']={'status':'SIMULATED_MEASURED','singular_values':s.tolist(),'C_K_max30':(u[:,:30]*s[:30]).tolist(),
                              'K11_privileged':False,'scientific_K_selection':'separate held-out six-task challenge'}
    labels=(x[:,:,0].mean(1)>0).astype(int)+2*(x[:,:,2].mean(1)>.5).astype(int);transition=np.zeros((4,4),int)
    for a,b in zip(labels[:-1],labels[1:]):transition[a,b]+=1
    layers['state_transition']={'status':'SIMULATED_MEASURED','partition':'mean electrode sign x mean spike count >0.5','counts':transition.tolist()}
    layers['attractor_basin']={'status':'FINITE_HORIZON_RECOVERY_PROXY','distance_to_preperturbation':np.linalg.norm(x[:,:,:2]-x[7,:,:2],axis=(1,2)).tolist(),
                             'asymptotic_attractor_proven':False}
    layers['spike_train']['npu_events']=record.get('spike_events',[])
    layers['spike']={'status':'SIMULATED_MEASURED','final_binary_events':[row[2] for st in states for row in st['n4']]}
    layers['sensorimotor'].update(kimi_action_integration='frozen Kimi cognitive action + mean NPU output')
    result['status']='LOCKED_SYNTHETIC_CARTOGRAPHY';return result
