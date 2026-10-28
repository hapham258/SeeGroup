from dataset import LayeredDepth

hf_cache_dir = None

layereddepth_dataset=dict(
    type=LayeredDepth,
    mode='val',
    hf_dataset='princeton-vl/LayeredDepth',
    hf_split='validation',
    streaming=False,
    cache_dir="dataset/splits",
    tuple_layer='layer_all',
    num_examples=300,
)
