(ns net.clojars.behrica.cluster_eval)

(import '[es.us.indices KMeansIndices]
        '[weka.core Attribute Instances DenseInstance EuclideanDistance]
        '[es.us.indices Cluster KMeansIndices])

(defn make-attributes [n]
  (map
   #(Attribute. (str "a-" %))
   (range n)))


(defn make-clusters [cluster-data]
  (let [
        instances (Instances. "test"
                              (java.util.ArrayList. (make-attributes (count (first (:values cluster-data)))))
                              (int 100))

        clusters
        (zipmap
         (distinct (:cluster cluster-data))
         (repeatedly 10 #(Cluster.)))

        _
        (mapv
         (fn [values cluster centroid?]

           (.add instances (DenseInstance. 1.0 (double-array values)))
           (.. (get clusters cluster) getInstances (add (.get instances
                                                              (dec (.size instances)))))
           (when centroid?
             (.setCentroide (get clusters cluster) (.get instances
                                                         (dec (.size instances))))))

         (:values cluster-data)
         (:cluster cluster-data)
         (:centroid? cluster-data))]

    (def clusters clusters)
    {:distance-fn (EuclideanDistance. instances)
     :clusters (vals clusters)}))



(defn cluster-index [cluster-data index-name]
  (let [clusters-result (make-clusters cluster-data)]
    (-> (clojure.lang.Reflector/invokeStaticMethod
         KMeansIndices "calcularSilhouette"
         (object-array [(clusters-result :clusters) (clusters-result :distance-fn)]))
        bean
        :resultado)))


(cluster-index
 {:values [[0 1 2] [2 3 4] [10 20 30] [30 40 20]]
  :cluster [0 0 1 1]
  :centroid? [false true false true]}
 "calcularSilhouette")