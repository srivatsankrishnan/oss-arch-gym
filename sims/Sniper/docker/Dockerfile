FROM ubuntu:18.04

WORKDIR /root/
RUN apt-get update
RUN apt-get -y install binutils build-essential curl git libboost-dev libbz2-dev libsqlite3-dev python wget zlib1g-dev
#RUN apt-get -y install libc6:i386 libncurses5:i386 libstdc++6:i386
#RUN dpkg --add-architecture i386

# Get Pinplay tool.
RUN curl -O https://www.intel.com/content/dam/develop/external/us/en/protected/pinplay-drdebug-3.7-pin-3.7-97619-g0d0c92f4f-gcc-linux.tar.gz
RUN tar xf pinplay-drdebug-3.7-pin-3.7-97619-g0d0c92f4f-gcc-linux.tar.gz
RUN rm pinplay-drdebug-3.7-pin-3.7-97619-g0d0c92f4f-gcc-linux.tar.gz

# Get Sniper.
ARG sniper_git
RUN git clone $sniper_git
WORKDIR /root/sniper
RUN ln -s /root/pinplay-drdebug-3.7-pin-3.7-97619-g0d0c92f4f-gcc-linux pin_kit
RUN make

ENV PATH=/root/sniper:$PATH

CMD ["run-sniper"]
