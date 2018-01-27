from os import listdir, path
from random import shuffle
import torch


class UserItemRecDataProviderAPI:
  def __init__(self, params, user_id_map, item_id_map):
    self._params = params
    self._data_dict = self.params['data_dict']
    self._i_id = 0 if 'itemIdInd' not in self.params else self.params['itemIdInd']
    self._u_id = 1 if 'userIdInd' not in self.params else self.params['userIdInd']
    self._r_id = 2 if 'ratingInd' not in self.params else self.params['ratingInd']
    self._major = 'items' if 'major' not in self.params else self.params['major']
    if not (self._major == 'items' or self._major == 'users'):
      raise ValueError("Major must be 'users' or 'items', but got {}".format(self._major))

    self._major_ind = self._i_id if self._major == 'items' else self._u_id
    self._minor_ind = self._u_id if self._major == 'items' else self._i_id
    self.user_id = 0  if 'user_id' not in self.params else self.params['user_id']

    #Map has to be defined
    self._user_id_map = user_id_map
    self._item_id_map = item_id_map
    
    major_map = self._item_id_map if self._major == 'items' else self._user_id_map
    minor_map = self._user_id_map if self._major == 'items' else self._item_id_map
    self._vector_dim = len(minor_map)
    self._batch_size = self.params['batch_size']

    self.data = dict()
    self.data[self.user_id] = []
    for item, rating in self._data_dict.items():   
      self.data[self.user_id].append((item, rating))


  def iterate_one_epoch(self):
    data = self.data
    keys = list(data.keys())
    shuffle(keys)
    s_ind = 0
    e_ind = self._batch_size
    while e_ind < len(keys):
      local_ind = 0
      inds1 = []
      inds2 = []
      vals = []
      for ind in range(s_ind, e_ind):
        inds2 += [v[0] for v in data[keys[ind]]]
        inds1 += [local_ind]*len([v[0] for v in data[keys[ind]]])
        vals += [v[1] for v in data[keys[ind]]]
        local_ind += 1

      i_torch = torch.LongTensor([inds1, inds2])
      v_torch = torch.FloatTensor(vals)

      mini_batch = torch.sparse.FloatTensor(i_torch, v_torch, torch.Size([self._batch_size, self._vector_dim]))
      s_ind += self._batch_size
      e_ind += self._batch_size
      yield  mini_batch

  def iterate_one_epoch_eval(self, for_inf=False):
    keys = list(self.data.keys())
    s_ind = 0
    while s_ind < len(keys):
      inds1 = [0] * len([v[0] for v in self.data[keys[s_ind]]])
      inds2 = [v[0] for v in self.data[keys[s_ind]]]
      vals = [v[1] for v in self.data[keys[s_ind]]]

      src_inds1 = [0] * len([v[0] for v in self.src_data[keys[s_ind]]])
      src_inds2 = [v[0] for v in self.src_data[keys[s_ind]]]
      src_vals = [v[1] for v in self.src_data[keys[s_ind]]]

      i_torch = torch.LongTensor([inds1, inds2])
      v_torch = torch.FloatTensor(vals)

      src_i_torch = torch.LongTensor([src_inds1, src_inds2])
      src_v_torch = torch.FloatTensor(src_vals)

      mini_batch = (torch.sparse.FloatTensor(i_torch, v_torch, torch.Size([1, self._vector_dim])),
                    torch.sparse.FloatTensor(src_i_torch, src_v_torch, torch.Size([1, self._vector_dim])))
      s_ind += 1
      if not for_inf:
        yield  mini_batch
      else:
        yield mini_batch, keys[s_ind - 1]

  @property
  def vector_dim(self):
    return self._vector_dim

  @property
  def userIdMap(self):
    return self._user_id_map

  @property
  def itemIdMap(self):
    return self._item_id_map

  @property
  def params(self):
    return self._params


