(ns scicloj.ml.cluster-eval)
; https://dzone.com/articles/kmeans-silhouette-score-explained-with-python-exam
(import '[moa.evaluation SilhouetteCoefficient]
        '[moa.cluster Cluster Clustering SphereCluster Miniball]
        '[com.yahoo.labs.samoa.instances InstanceImpl DenseInstance DenseInstanceData InstancesHeader Attribute Instances]
        '[moa.gui.visualization DataPoint])


(defn make-miniball [vs]
  (let [mb (Miniball. (count (first vs)))]
    (run!
     #(.check_in mb (double-array %)) vs)
    (.build mb)
    {:radius (.radius mb)
     :center (.center mb)
     :d (double (count (first vs)))}))


(defn make-datapoint [v ih]
  (let  [inst (InstanceImpl. 1.0 (double-array v))
         _ (.setDataset inst ih)]

    (DataPoint. inst  (int 0))))

(def vss
  [[[1 3 4]
    [2 5 6]]
   [[2 4 6]
    [3 7 5]]])



(defn silhouette-coefficient [vss]
  (let [
        dimensions (count (first (first vss)))
        ih (InstancesHeader.
            (Instances. "test" (java.util.ArrayList.
                                (map #(Attribute. (str "feature-" %)) (range dimensions)))
                        dimensions))

        mbs (map  make-miniball vss)
        scs (map #(SphereCluster. (:center %)
                                  (:radius %)
                                  (:d %))
                 mbs)
        clustering (Clustering. (into-array Cluster scs))
        evaluation-points (java.util.ArrayList. (map #(make-datapoint % ih) (apply concat vss)))
        sc (SilhouetteCoefficient.)
        _ (.evaluateClustering sc clustering nil evaluation-points)]
    ;; (.getNames sc)
    {:data-points evaluation-points
     :normalized-silhouette-coefficient
     (first  (.getAllValues sc 0))}))


(comment
  (require '[tablecloth.api :as tc]
           '[fastmath.clustering :as clust])

  (def iris
    (->
     (tc/dataset
      "https://raw.githubusercontent.com/scicloj/metamorph.ml/main/test/data/iris.csv" {:key-fn keyword})))

  (def res
    (clust/k-means (-> iris (tc/select-columns :type/float ) (tc/rows :as-vec)) 5))


  (def vss
    (->>
     (map #(hash-map :clustering %2 :data %1)

          (get res :data)
          (get res :clustering))
     (group-by :clustering)
     (map (fn [groups] (map :data (second  groups))))))

  (def sil
    (silhouette-coefficient vss))



  :ok)

(comment
  (def data
    (tc/append
     (->
      (tc/dataset "points_x.csv")
      (tc/select-columns ["0" "1"])
      (tc/rename-columns {"0" :x-0 "1" :x-1}))
     (->
      (tc/dataset "points_y.csv")
      (tc/select-columns ["0"])
      (tc/rename-columns ["0"] {"0" :y}))))

  (def vss
    (map
     #(tc/rows % :as-vec)
     (-> data
         (tc/group-by :y)
         (tc/select-columns [:x-0 :x-1])
         (tc/groups->seq))))

  (silhouette-coefficient vss)




  :ok)
