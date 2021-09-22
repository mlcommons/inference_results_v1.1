import tensorflow as tf
f = tf.io.gfile.GFile("int8_resnet50_v1.pb", 'rb')
gd = tf.compat.v1.GraphDef()
gd.ParseFromString(f.read())
with tf.Graph().as_default() as graph:
    tf.import_graph_def(gd, name="")

print(type(graph))

graph.get_operations()

import graphsurgeon as gs
dynamic_graph = gs.DynamicGraph(graph)
node = dynamic_graph.find_nodes_by_name('ArgMax')[0]
node.attr["output_type"].type = 1 # FP32
dynamic_graph.write("int8_resnet50_v1_fp32.pb")

