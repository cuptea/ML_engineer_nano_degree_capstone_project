import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
  
class NCF(nn.Module):
    """
    This is a custom implementation of Neural Colaborative Filtering
    msno                              4112
    song_id                         295930
    source_system_tab                    1
    source_screen_name                   8
    source_type                          1
    target                               1
    city                                10 
    gender                               2
    registered_via                       1
    age                                  0
    app_age                           2103
    registration                      2838
    song_length                     206471
    composer                        187514
    lyricist                             0
    language                             6
    year                              2016
    country                            143
    genre                              156
    artist                          220753
    """
    def __init__(self,
                 n_msno= 34403,
                 n_song_id= 2296869,
                 n_source_system_tab=9,
                 n_source_screen_name=21,
                 n_source_type=13,
                 n_city=22,
                 n_gender=4,
                 n_registered_via=7,
                 n_composer=360936,
                 n_lyricist=128374,
                 n_language=12,
                 n_country=201,
                 n_genre=192,
                 n_artist=239750,
                 n_dim=10):
        
        """
        Initialize the model by settingg up the various layers.
        """
        super(NCF, self).__init__()
        
        self.msno_nn_embed=nn.Embedding(n_msno, min(n_dim,n_msno), sparse=False)
        self.msno_mf_embed=nn.Embedding(n_msno, n_dim, sparse=False)
        self.msno_mf_bias=nn.Embedding(n_msno, 1, sparse=False)
        
        self.song_id_nn_embed=nn.Embedding(n_song_id, min(n_dim,n_song_id), sparse=False)
        self.song_id_mf_embed=nn.Embedding(n_song_id, n_dim, sparse=False)
        self.song_id_mf_bias=nn.Embedding(n_song_id, 1, sparse=False)
        
        self.source_system_tab_embed=nn.Embedding(n_source_system_tab, min(n_dim,n_source_system_tab), sparse=False)
        self.source_screen_name_embed=nn.Embedding(n_source_screen_name, min(n_dim,n_source_screen_name), sparse=False)
        self.source_type_embed=nn.Embedding(n_source_type, min(n_dim,n_source_type), sparse=False)
        self.city_embed=nn.Embedding(n_city, min(n_dim,n_city), sparse=False)
        self.gender_embed=nn.Embedding(n_gender, min(n_dim,n_gender), sparse=False)
        self.registered_via_embed=nn.Embedding(n_registered_via, min(n_dim,n_registered_via), sparse=False)
        self.composer_embed=nn.Embedding(n_composer, min(n_dim,n_composer), sparse=False)
        self.lyricist_embed=nn.Embedding(n_lyricist, min(n_dim,n_lyricist), sparse=False)
        self.language_embed=nn.Embedding(n_language, min(n_dim,n_language), sparse=False)
        self.country_embed=nn.Embedding(n_country, min(n_dim,n_country), sparse=False)
        self.genre_embed=nn.Embedding(n_genre, min(n_dim,n_genre), sparse=False)
        self.artist_embed=nn.Embedding(n_artist, min(n_dim,n_artist), sparse=False)
        
        self.concat_dim = min(n_dim,n_msno)\
                        + min(n_dim,n_song_id)\
                        + min(n_dim,n_source_system_tab)\
                        + min(n_dim,n_source_screen_name)\
                        + min(n_dim,n_source_type)\
                        + min(n_dim,n_city)\
                        + min(n_dim,n_gender)\
                        + min(n_dim,n_registered_via)\
                        + min(n_dim,n_composer)\
                        + min(n_dim,n_lyricist)\
                        + min(n_dim,n_language)\
                        + min(n_dim,n_country)\
                        + min(n_dim,n_genre)\
                        + min(n_dim,n_artist)
                    
        self.l1 = torch.nn.Linear(self.concat_dim, self.concat_dim*2)
        self.l2 = torch.nn.Linear(self.concat_dim*2, self.concat_dim)
        self.l3 = torch.nn.Linear(self.concat_dim, n_dim)
        
        self.l4 = torch.nn.Linear(2*n_dim, 1)
        self.sig = nn.Sigmoid()
        
    def forward(self, x):
        msno = x['msno'].long()
        song_id = x['song_id'].long()
        source_system_tab = x['source_system_tab'].long()
        source_screen_name= x['source_screen_name'].long()
        source_type = x['source_type'].long()
        city = x['city'].long()
        gender = x['gender'].long()
        registered_via = x['registered_via'].long()
        composer = x['composer'].long()
        lyricist = x['lyricist'].long()
        language = x['language'].long()
        country = x['country'].long()
        genre = x['genre'].long()
        artist = x['artist'].long()

        concat_embed = torch.cat((self.msno_nn_embed(msno),
                                  self.song_id_nn_embed(song_id),
                                  self.source_system_tab_embed(source_system_tab),
                                  self.source_screen_name_embed(source_screen_name),
                                  self.source_type_embed(source_type),
                                  self.city_embed(city),
                                  self.gender_embed(gender),
                                  self.registered_via_embed(registered_via),
                                  self.composer_embed(composer),
                                  self.lyricist_embed(lyricist),
                                  self.language_embed(language),
                                  self.country_embed(country),
                                  self.genre_embed(genre),
                                  self.artist_embed(artist)), 1)

        mf_embed = torch.mul(self.msno_mf_embed(msno), self.song_id_mf_embed(song_id))
        
        h1 = torch.relu(self.l1(concat_embed))
        h2 = torch.relu(self.l2(h1))
        h3 = torch.relu(self.l3(h2))
        
        logit = self.l4(torch.cat([mf_embed, h3], dim=-1))
        
        msno_bias = self.msno_mf_bias(msno)
        song_bias = self.song_id_mf_bias(song_id)
        
        logit = logit + self.song_id_mf_bias(song_id)+ self.msno_mf_bias(msno)
      
        
        return self.sig(logit)
  