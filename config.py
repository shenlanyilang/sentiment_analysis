# -*- coding:utf-8 -*-

config = {
    'num_class': 4,
    'seq_len': 250,
    'num_filters': 256,
    'filter_sizes': [3,4,5,6],
    'train_file': './data/train.csv',
    'val_file': './data/valid.csv',
    'w2v_path': './data/sm_word_emb.wiki.w2v',
    'feature_column': 'content',
    'labels': ['location_traffic_convenience',
              'location_distance_from_business_district',
              'location_easy_to_find',
              'service_wait_time', 'service_waiters_attitude',
              'service_parking_convenience', 'service_serving_speed',
              'price_level','price_cost_effective', 'price_discount',
              'environment_decoration', 'environment_noise',
              'environment_space', 'environment_cleaness','dish_portion',
              'dish_taste', 'dish_look', 'dish_recommendation',
              'others_overall_experience', 'others_willing_to_consume_again'
               ]
}
