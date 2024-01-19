# Copyright (c) Paolo Dimasi, Politecnico di Torino.
from pandaset import DataSet
import os, time, shutil
import sys, getopt
from rdfutils import RDFSerialization
from rdflib import URIRef,  Literal, Namespace,  XSD



dataset_path = '' 
dataset_name = 'prior_kg/raw/pandaset_kg'
try:
    opts, args = getopt.getopt(sys.argv[1:],"f:d:",["folder=", "dataset="])
except getopt.GetoptError:
    print("pandaset2RDF.py -f <input-folder>")
    sys.exit(2)
for opt, arg in opts:
    if opt in ("-f", "--folder"):
        dataset_path = arg
        
map_key ={ 'uuid': 1,'label':2,  'yaw':3, 'stationary':4, 
          'position.x': 6,'position.y': 7,'position.z': 8,
          'dimensions.x': 9,'dimensions.y': 10,'dimensions.z': 11,
          'attributes.object_motion':12,  
          # found doc error on pandaset doc
          'attributes.rider_status':[15,16],'attributes.pedestrian_behavior': 15}

num_sequence= 0
stats = {
    'sequence' :{},
    'category':{},
    'relationship':{ }
}        




try:
    os.mkdir('prior_kg/raw')
except:
    shutil.rmtree('prior_kg/raw')
    os.mkdir('prior_kg/raw')

panda_ds = DataSet(dataset_path)



print("Knowledge graph generation...")
processTime = time.process_time()
actualTime = time.time()

# graph creation 
NS = Namespace('http://pandaset.org/')
serialize = RDFSerialization(dataset_name, 'panda', NS)



# loop over all sequence (dynamic scene concept )
for id_seq in panda_ds.sequences():

    # loop over data and create
    seq = panda_ds[id_seq]
    seq.load_cuboids()
    
    num_sequence+=1
    

    for i,cub in enumerate(seq.cuboids.data):
        
        if id_seq not in stats['sequence'].keys():
           stats['sequence'][id_seq]=1
        else:
            stats['sequence'][id_seq]+=1
        
        for cuboid in cub.itertuples():
            to_save =False
            uuid = cuboid[map_key['uuid']]
            category = str(cuboid[map_key['label']])
            yaw = cuboid[map_key['yaw']]
            dim_x = cuboid[map_key['dimensions.x']]
            dim_y = cuboid[map_key['dimensions.y']]
            dim_z = cuboid[map_key['dimensions.z']]
            pos_x = cuboid[map_key['position.x']]
            pos_y = cuboid[map_key['position.y']]
            pos_z = cuboid[map_key['position.z']]
            #static= cuboid[map_key['stationary']]
            
            if  category  in ['Trolley', 'Tram / Subway', 'Towed Object',
                            'Personal Mobility Device', 'Trolley',
                            'Other Vehicle - Pedicab', 'Other Vehicle - Uncommon',
                            'Motorized Scooter'] or category.startswith('Animal'):
                category  = 'Dynamic'
                
            elif category  in ['Pylons','Cones', 'Rolling Containers'] or category.endswith('Barriers'):
                category = 'Static' if not category.endswith('Barriers') else 'Fence'
                
                state = "in/on"
                serialize.rdf_graph.add((URIRef(NS[category]), URIRef(NS[state]), URIRef(NS['Road'])))

                if state not in stats['relationship'].keys():
                    stats['relationship'][state]=1
                else:
                    stats['relationship'][state]+=1
                
                obj = 'Road'
                if obj not in stats['category'].keys():
                    stats['category'][obj]=1
                else:
                    stats['category'][obj]+=1
                    
                to_save =True
                
            elif category.endswith('Truck')  or category.endswith('truck'):
                category = 'Truck'
            elif category.endswith('Construction Vehicle') or category.endswith('Emergency Vehicle'):
                category='Caravan'
                
            elif category.startswith('Pedestrian'):
                
                if category.endswith('Object'):
                    serialize.rdf_graph.add((URIRef(NS['Person']), URIRef(NS["with"]), URIRef(NS['Object'])))
                  
                    state = 'with'
                    if state not in stats['relationship'].keys():
                        stats['relationship'][state]=1
                    else:
                        stats['relationship'][state]+=1
                    
                    obj = 'Object'
                    if obj not in stats['category'].keys():
                        stats['category'][obj]=1
                    else:
                        stats['category'][obj]+=1
                    
                    to_save =True
                    
                    
                category = 'Person'
                # print(cuboid[:])
                state = str(cuboid[map_key['attributes.pedestrian_behavior']])
                if state in ['Sitting', 'Lying', 'Walking','Standing']:
                    serialize.rdf_graph.add((URIRef(NS['Person']), URIRef(NS['hasState']), URIRef(NS[state])))
                   
                    
                    if state not in stats['relationship'].keys():
                        stats['relationship'][state]=1
                    else:
                        stats['relationship'][state]+=1
                        
                    to_save =True
            elif category.find('Sign')>=0:
                category = 'TrafficSign' 
                    
            category = category.replace(' ', '')

            # State
            if category in [ 'Car', 'Caravan', 'Truck', 'Motorcycle','Bicycle', 'Bus', 'Dynamic'] \
                    and cuboid[map_key['attributes.object_motion']] in ['Parked', 'Stopped', 'Moving']:
                state = str( cuboid[map_key['attributes.object_motion']] )
                serialize.rdf_graph.add((URIRef(NS[f'CategoryId_{uuid}']), URIRef(NS["hasState"]), URIRef(NS[state])))
              
                
                to_save =True
                
                
                if state not in stats['relationship'].keys():
                    stats['relationship'][state]=1
                else:
                    stats['relationship'][state]+=1
            
            # Rider status 
            if category in [ 'Motorcycle','Bicycle']:
                state = list(filter(lambda x: x in ['With Rider', 'Without Rider'], cuboid[13:]) )
           
            
                if  len(state)>0:
                        
                    state = state[0].split(' ')[0].lower()
                    
                    serialize.rdf_graph.add((URIRef(NS[f'CategoryId_{uuid}']), URIRef(NS[state]), URIRef(NS['Rider'])))
                    
                    if state not in stats['relationship'].keys():
                        stats['relationship'][state]=1
                    else:
                        stats['relationship'][state]+=1
                        
                    if state not in stats['category'].keys():
                        stats['category']['Rider']=1
                    else:
                        stats['category']['Rider']+=1
                        
                    to_save =True
                    
            if to_save==True:
                if category not in stats['category'].keys():
                    stats['category'][category]=1
                else:
                    stats['category'][category]+=1
                    
                
                serialize.rdf_graph.add((URIRef(NS[f'Sequence_{id_seq}']), URIRef(NS["hasParticipant"]), URIRef(NS[f'CategoryId_{uuid}'])))             
                
                serialize.rdf_graph.add((URIRef(NS[f'CategoryId_{uuid}']),  URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"), URIRef(NS[category])))             
               
                # Translatiom
        
                serialize.rdf_graph.add((URIRef(NS[f'CategoryId_{uuid}']), URIRef(NS[f"hasTraslationX"]), Literal(dim_x, datatype=XSD.double)))
                serialize.rdf_graph.add((URIRef(NS[f'CategoryId_{uuid}']), URIRef(NS[f"hasTraslationY"]), Literal(dim_y, datatype=XSD.double)))
                serialize.rdf_graph.add((URIRef(NS[f'CategoryId_{uuid}']), URIRef(NS[f"hasTraslationZ"]), Literal(dim_z, datatype=XSD.double)))
              
                
                if 'translation' not in stats['relationship'].keys():
                    stats['relationship']['translation']=3
                else:
                    stats['relationship']['translation']+=3
                # Size

                serialize.rdf_graph.add((URIRef(NS[f'CategoryId_{uuid}']), URIRef(NS[f"hasPosX"]), Literal(pos_x, datatype=XSD.double)))
                serialize.rdf_graph.add((URIRef(NS[f'CategoryId_{uuid}']), URIRef(NS[f"hasPosY"]), Literal(pos_y, datatype=XSD.double)))
                serialize.rdf_graph.add((URIRef(NS[f'CategoryId_{uuid}']), URIRef(NS[f"hasPosZ"]), Literal(pos_z, datatype=XSD.double)))
              
                
                if 'size' not in stats['relationship'].keys():
                    stats['relationship']['size']=3
                else:
                    stats['relationship']['size']+=3
                    
                # Rotation
                serialize.rdf_graph.add(( URIRef(NS[f'CategoryId_{uuid}']), URIRef(NS[f"hasRotationZ"]), Literal(yaw, datatype=XSD.double) ))
                
                if 'rotation' not in stats['relationship'].keys():
                    stats['relationship']['rotation']=1
                else:
                    stats['relationship']['rotation']+=1
            
            serialize.batch_check()

            
# write final triples 
serialize.write()
        

print("Processing time: ", (time.process_time() - processTime))
print("Absolut time (Minutes):    ", ((time.time() - actualTime)/60))

with open('prior_kg/pandaset_kg_stats.txt', 'w') as f:
    f.write(f'number of sequence:{num_sequence}\n')
    f.writelines([ f'{k1}\t{k2}\t{stats[k1][k2]}\n'  for k1 in stats.keys() for k2 in stats[k1].keys() ] )
                