(ns scicloj.ml.top2vec-test
  (:require [clojure.test :refer :all]
            [scicloj.ml.top2vec :refer :all]
            [camel-snake-kebab.core :as csk]
            [tablecloth.api :as tc]))


(deftest test-train []
  (def raw-data
    (tc/dataset "https://github.com/scicloj/scicloj.ml.smile/blob/main/test/data/reviews.csv.gz?raw=true"
                {:key-fn csk/->kebab-case-keyword
                 :file-type :csv
                 :gzipped? true}))
  (def data
    (-> raw-data
        (tc/shuffle {:seed 123})
        (tc/head 10000)
        (tc/select-columns :text)
        tc/drop-missing))

  (def train-result-learn
    (scicloj.metamorph.ml/train data {:speed :learn
                                      :model-type :top2vec
                                      :pwi
                                      :min_count 1
                                      :documents-column :text})))