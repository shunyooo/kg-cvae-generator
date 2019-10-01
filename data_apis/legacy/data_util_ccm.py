from torch.utils.data import Dataset
import torch
import pprint


class CCMDataset(Dataset):
    @staticmethod
    def padding(sent, max_len):
        return sent + ['_EOS'] + ['_PAD'] * (max_len - len(sent) - 1)

    def padding_triple(self, triple, num, max_len):
        newtriple = []
        triple = [[self.NAF]] + triple
        for tri in triple:
            newtriple.append(
                tri + [['_PAD_H', '_PAD_R', '_PAD_T']] * (max_len - len(tri)))
        pad_triple = [['_PAD_H', '_PAD_R', '_PAD_T']] * max_len
        return newtriple + [pad_triple] * (num - len(newtriple))

    def __init__(self, data, csk_entities, csk_triples, is_train,
                 vocab_indexer, entity_indexer):
        super(CCMDataset).__init__()
        self.vocab_indexer = vocab_indexer
        self.entity_indexer = entity_indexer

        self.NAF = ['_NAF_H', '_NAF_R', '_NAF_T']

        encoder_len = max([len(item['post']) for item in data]) + 1
        decoder_len = max([len(item['response']) for item in data]) + 1
        triple_num = max([len(item['all_triples']) for item in data]) + 1
        triple_len = max([len(tri) for item in data for tri in item['all_triples']])

        self.encoder_len = encoder_len
        self.decoder_len = decoder_len
        self.triple_num = triple_num
        self.triple_len = triple_len

        posts, responses, posts_length, responses_length = [], [], [], []
        entities, triples, matches, post_triples, response_triples = [], [], [], [], []
        match_triples, all_triples = [], []

        self.data = []
        for item in data:
            data_item = {}
            data_item['posts_length'] = [len(item['post']) + 1]
            data_item['_post'] = self.padding(item['post'], encoder_len)
            data_item['post'] = vocab_indexer.encode(data_item['_post'])
            data_item['post_entity'] = entity_indexer.encode(data_item['_post'])

            data_item['responses_length'] = [len(item['response']) + 1]
            data_item['_response'] = self.padding(item['response'], decoder_len)
            data_item['response'] = vocab_indexer.encode(data_item['_response'])

            padded_triple = self.padding_triple(
                [[csk_triples[x].split(', ') for x in triple] for triple in
                 item['all_triples']], triple_num, triple_len)

            data_item['_triples'] = padded_triple
            data_item['triples'] = [[entity_indexer.encode(triple) for triple in triple_lists] for triple_lists in data_item['_triples']]

            if not is_train:
                entity = [['_NONE'] * triple_len]
                for ent in item['all_entities']:
                    entity.append([csk_entities[x] for x in ent] + ['_NONE'] * (
                            triple_len - len(ent)))
                entity = entity + [['_NONE'] * triple_len] * (triple_num - len(entity))

            else:
                entity = [['_NONE'] * triple_len] * triple_num

            data_item['_entity'] = entity
            data_item['entity'] = [entity_indexer.encode(entity) for entity in data_item['_entity']]

            post_triple = ([[x] for x in item['post_triples']] + [[0]] * (encoder_len - len(item['post_triples'])))
            data_item['post_triple'] = post_triple

            response_triple = ([self.NAF] +
                               [self.NAF if x == -1 else csk_triples[x].split(', ') for x in item['response_triples']] +
                               [self.NAF] * (decoder_len - 1 - len(item['response_triples'])))
            data_item['response_triple'] = [entity_indexer.encode(entity) for entity in response_triple]

            match_index = []
            for idx, val in enumerate(item['match_index']):
                _index = [-1] * triple_num
                if val[0] == -1 and val[1] == -1:
                    match_index.append(_index)
                else:
                    _index[val[0]] = val[1]
                    t = data_item['triples'][val[0]][val[1]]
                    assert (t == data_item['response_triple'][idx + 1])
                    match_index.append(_index)
            data_item['match_triples'] = match_index + [[-1]*triple_num]*(decoder_len - len(match_index))

            self.data.append(data_item)

    def get_maxlen(self):
        return self.encoder_len, self.decoder_len, self.triple_num, self.triple_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
        target_list = ['posts_length', 'entity', 'post_entity', 'post',
                       'responses_length', 'response', 'all_triples', 'triples',
                       'post_triple', 'response_triple', 'match_triples']
        data_dict = {key: torch.LongTensor(value) for key, value in data_item.items()
                     if key in target_list}
        #print(data_dict)
        return data_dict