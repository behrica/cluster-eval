(ns build
  (:require [clojure.tools.build.api :as b]))

(def lib 'net.clojars.behrica/cluster_eval)
(def version (format "0.1.%s" (b/git-count-revs nil)))
(def class-dir "target/classes")
(def basis (b/create-basis {:project "deps.edn"}))
(def jar-file (format "target/%s-%s.jar" (name lib) version))

(defn clean [_]
  (b/delete {:path "target"}))
(defn compile [_]
  (b/javac {:src-dirs ["java"]
            :class-dir class-dir
            :basis basis}))
            ;; :javac-opts ["-source" "8" "-target" "8"]


(defn jar [_]
  (compile nil)
  (b/write-pom {:class-dir class-dir
                :lib lib
                :version version
                :basis basis
                :src-dirs ["src"]})
  (b/copy-dir {:src-dirs ["src" "resources"]
               :target-dir class-dir})
  (b/jar {:class-dir class-dir
          :jar-file jar-file}))

(defn install [_]
  (b/install {:basis basis
              :lib lib
              :version version
              :class-dir class-dir
              :jar-file jar-file})
  (println "Installed in .m2: " jar-file))

(defn ci [_]
  (clean nil)
  (jar nil)
  (install nil))
