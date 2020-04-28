IMG:=`cat Name`
GPU_OPT:=all
MOUNT_ETC:=
MOUNT_LOG:=
MOUNT_APP:=-v `pwd`/work/app:/app
MOUNT:=$(MOUNT_ETC) $(MOUNT_LOG) $(MOUNT_APP)
EXT_VOL:=
PORT_MAP:=
LINK_MAP:=
RESTART:=no
CONTAINER_NAME:=docker-cuda9-py36-hello

echo:
	echo $(IMG)

run:
	docker rm $(CONTAINER_NAME) || echo
	docker run -d --gpus $(GPU_OPT) --name $(CONTAINER_NAME) $(LINK_MAP) $(PORT_MAP) --restart=$(RESTART) \
                         $(EXT_VOL) $(MOUNT) $(IMG)

run_i:
	docker rm $(CONTAINER_NAME) || echo
	docker run -i -t --gpus $(GPU_OPT) --name $(CONTAINER_NAME) $(LINK_MAP) $(PORT_MAP) \
                         $(EXT_VOL) $(MOUNT) $(IMG) /bin/bash 

exec_i:
	docker exec -i -t --name $(CONTAINER_NAME)  /bin/bash 

stop:
	docker stop $(CONTAINER_NAME)

rm:	stop
	docker rm $(CONTAINER_NAME)

