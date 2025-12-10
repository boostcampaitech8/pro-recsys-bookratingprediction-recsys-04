from .catboost_data import (
    catboost_data_load,
    catboost_data_split,
    catboost_data_loader,
)
# Feature Name	설명
# user_review_count	유저의 총 리뷰 수
# user_review_count_log	log scale 리뷰 수
# user_review_count_log_bin	리뷰 수 로그 bin
# user_genre_variety	유저가 본 장르 개수
# user_top_genre	최애 장르
# user_genre_entropy	취향 다양성 entropy
# user_author_variety	본 작가 다양성
# user_top_author	최애 작가
# user_top_publisher	최애 출판사
# user_mean_book_age	평균 book age
# location_cluster	지역 기반 클러스터