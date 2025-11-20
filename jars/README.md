# External jar files
Please download external jar files from the following sites and put them under this directory.


- [MASON](https://github.com/eclab/mason/releases/tag/v20) version 20
  - The original GMU download mirror is now offline. Grab the `v20` release from GitHub (zip or source), build it with `mvn clean install`, and copy `target/mason-20.jar` into this directory. Install it into your local Maven repo with `mvn install:install-file -DgroupId=cs.gmu.edu.eclab -DartifactId=mason -Dversion=20 -Dpackaging=jar -Dfile=./jars/mason-20.jar` so AMLSim's build can resolve it offline.
- [Commons-Math](http://commons.apache.org/proper/commons-math/download_math.cgi) version 3.6.1
  - Download "commons-math3-3.6.1-bin.tar.gz" and extract the following jar files.
    - commons-math3-3.6.1.jar
    - commons-math3-3.6.1-tests.jar
    - commons-math3-3.6.1-tools.jar
- [JSON in Java](https://jar-download.com/artifacts/org.json/json/20180813) version 20180813
  - The latest jar file is available from [here](https://github.com/stleary/JSON-java)
- [WebGraph](http://webgraph.di.unimi.it/) version 3.6.1
  - Please download "binary tarball" from the homepage and extract the jar file.
  - It is convenient to download "dependencies tarball" for the following dependencies. 
    - [DSI Utilities](http://dsiutils.di.unimi.it/) version 2.5.4
    - [fastutil](http://fastutil.di.unimi.it/) version 8.2.3
    - [Sux for Java](http://sux.di.unimi.it/) version 4.2.0
    - [JSAP](http://www.martiansoftware.com/jsap/) version 2.1
    - [SLF4J](https://www.slf4j.org/download.html) version 1.7.25
- [MySQL Connector for Java](https://dev.mysql.com/downloads/connector/j/5.1.html) version 5.1.48
- [JUnit5](https://search.maven.org/artifact/org.junit.platform/junit-platform-console-standalone/1.8.1/jar) version 5
- [Mockito Core](https://mvnrepository.com/artifact/org.mockito/mockito-core/4.0.0) version 4.0.0
  - Please download the following dependencies
    - [Byte Buddy](https://mvnrepository.com/artifact/net.bytebuddy/byte-buddy/1.11.19) version 1.11.19
    - [Byte Buddy Agent](https://mvnrepository.com/artifact/net.bytebuddy/byte-buddy-agent/1.11.19) version 1.11.19
    - [Objenesis](https://mvnrepository.com/artifact/org.objenesis/objenesis/3.2) version 3.2
- [Mockito Inline](https://mvnrepository.com/artifact/org.mockito/mockito-inline/4.0.0) version 4.0.0


