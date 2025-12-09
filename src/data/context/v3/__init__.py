# <피쳐엔지니어링>
# v1,v2 의 feature_importance 를 고려하여, 
# 가설1) 대형 출판사 일 수록 책 퀄 높 -> rating 높 -> 리뷰 많다 or 책 많다 : publisher_book_count
# 가설2) 유명 저자 일수록 -> rating 높& 낸 책 많을 것 -> : author_book_count 
# summary_cluster upgrad (p)
# book_title -> 구간화(not unique 하게)
# 출판사만 다른 책 처리 -> canonical_id 생성



from .catboost_data import (
    catboost_data_load,
    catboost_data_split,
    catboost_data_loader,
)