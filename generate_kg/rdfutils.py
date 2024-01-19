# Copyright (c) Paolo Dimasi, Politecnico di Torino.
from rdflib import  Namespace, Graph

# RDF Serialization 

MAX_NUM_TRIPLE = 10000


class RDFSerialization():
    def __init__(self, dataset_name: str, prefix:str, namespace: Namespace ):
        self.rdf_graph = Graph()
        self.rdf_graph.bind(prefix, namespace)
        self._number = 0
        self._prefix = prefix
        self._namespace = namespace
      
        self._dataset_name = dataset_name
    

    def write(self):
    
        if len(self.rdf_graph)>0:
            print(f"serializing ... ('{self._dataset_name}_{self._number}.nt')")
            self.rdf_graph.serialize(destination=f'{self._dataset_name}_{self._number}.nt',  format='nt', encoding='utf-8',)
            self.rdf_graph = Graph()
            self.rdf_graph.bind(self._prefix, self._namespace)
            
            self._number += 1
        else:
            print('No data to serialize up to this point!')
        
        
    def batch_check(self):
        if len(self.rdf_graph) > MAX_NUM_TRIPLE :
            self.write()
            
    
    @property
    def number_item(self):
        return self._number

        
        