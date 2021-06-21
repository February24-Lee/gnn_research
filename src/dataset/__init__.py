from .CiteSeer      import citeSeer,    citeSeer_pl
from .Cora_ml       import cora_ml,     cora_ml_pl
from .Pubmed        import pubmed,      pubmed_pl
from .ArXiv         import arxiv,       arxiv_pl
from .Flicker       import flickr,      flicker_pl
from .Products      import products,    products_pl


DATASET_LIST = {'citeseer'  : citeSeer_pl,
                'core_ml'   : cora_ml_pl,
                'pumbed'    : pubmed_pl,
                'arxiv'     : arxiv_pl,
                'flicker'   : flicker_pl,
                'products'  : products_pl}