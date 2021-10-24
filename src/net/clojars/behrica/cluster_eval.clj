(ns net.clojars.behrica.cluster_eval
  (:import [es.us.indices KMeansIndices]
          [weka.core Attribute Instances DenseInstance EuclideanDistance]
          [es.us.indices Cluster KMeansIndices]))

(defn make-attributes [n]
  (map
   #(Attribute. (str "a-" %))
   (range n)))


(defn make-clusters [cluster-data]
  (let [
        _ (def cluster-data cluster-data)
        instances (Instances. "test"
                              (java.util.ArrayList. (make-attributes (count (first (:values cluster-data)))))
                              (int 100))

        clusters
        (zipmap
         (distinct (:cluster cluster-data))
         (repeatedly  #(Cluster.)))

        _
        (mapv
         (fn [values cluster centroid?]

           (let  [new-instance (DenseInstance. 1.0 (double-array values))
                  _ (.add instances new-instance)
                  current-cluster (get clusters cluster)
                  current-cluster-instances (.getInstances current-cluster)
                  _ (.setDataset new-instance instances)]


             (if centroid?
               (.setCentroide current-cluster new-instance)
               (.add current-cluster-instances new-instance))))
         (:values cluster-data)
         (:cluster cluster-data)
         (:centroid? cluster-data))]

    (def clusters clusters)
    {:distance-fn (EuclideanDistance. instances)
     :clusters (vals clusters)}))

(first (vals  clusters))

(defn cluster-index [cluster-data index-name]
  (let [clusters-result (make-clusters cluster-data)
        _ (def clusters-result clusters-result)]
    (-> (clojure.lang.Reflector/invokeStaticMethod
         KMeansIndices index-name
         (object-array [(clusters-result :clusters) (clusters-result :distance-fn)]))
        bean
        :resultado)))

(comment
  (cluster-index
   {:values [[0 1 2]  [2 3 4] [2 3 4] [10 20 30] [10 20 30] [30 40 20]]
    :cluster [1 0 1 1 1]
    :centroid? [false true false true false true]}
   "calcularSilhouette")

  (cluster-index
   {:values [[0 1 2] [2 3 4] [10 20 30] [30 40 20]]
    :cluster [1 0 1 2]
    :centroid? [false true false true]}
   "calcularSilhouette"))
