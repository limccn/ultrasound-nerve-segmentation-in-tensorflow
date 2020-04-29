IMG:=`cat Name`
GPU_OPT:=all
MOUNT_ETC:=
MOUNT_LOG:=
MOUNT_APP:=-v `pwd`/work/app:/home/devel/app
MOUNT:=$(MOUNT_ETC) $(MOUNT_LOG) $(MOUNT_APP)
EXT_VOL:=
PORT_MAP:=-p 10066:6000 -p 10065:5000
LINK_MAP:=
RESTART:=no
#RUN_USER:=--user root
RUN_USER:=
CONTAINER_NAME:=docker-cuda9-py36-uns
CMD_JUPYTER=jupyter notebook --allow-root --notebook-dir=/home/devel/app --no-browser --port 6000 --ip=0.0.0.0
#CMD_TEST_SCORE=python -u /app/nerve_test_score.py > /app/logs/test/score_`date +%Y%m%d%H%m%s`.log
CMD_TEST_SCORE=/bin/bash -c 'cd /app/test && python -u nerve_test_score.py'

echo:
	echo $(IMG)

run:
	docker rm $(CONTAINER_NAME) || echo
	docker run -d --gpus $(GPU_OPT) --name $(CONTAINER_NAME) $(LINK_MAP) $(PORT_MAP) --restart=$(RESTART) \
                         $(RUN_USER) $(EXT_VOL) $(MOUNT) $(IMG)

run_i:
	docker rm $(CONTAINER_NAME) || echo
	docker run -i -t --gpus $(GPU_OPT) --name $(CONTAINER_NAME) $(LINK_MAP) $(PORT_MAP) \
                         $(RUN_USER) $(EXT_VOL) $(MOUNT) $(IMG) /bin/bash 

run_jupyter:
	docker rm $(CONTAINER_NAME) || echo
	docker run -i -t --gpus $(GPU_OPT) --name $(CONTAINER_NAME) $(LINK_MAP) $(PORT_MAP) \
                         $(RUN_USER) $(EXT_VOL) $(MOUNT) $(IMG) $(CMD_JUPYTER)
	echo  $(CMD_JUPYTER_PASSWD)

run_test_score:
	docker rm $(CONTAINER_NAME) || echo
	docker run -i -t --gpus $(GPU_OPT) --name $(CONTAINER_NAME) $(LINK_MAP) $(PORT_MAP) \
                         $(RUN_USER) $(EXT_VOL) $(MOUNT) $(IMG) $(CMD_TEST_SCORE)

exec_i:
	docker exec -i -t --name $(CONTAINER_NAME)  /bin/bash 

exec_jupyter:
	docker exec -i -t --name $(CONTAINER_NAME)  $(CMD_JUPYTER)

exec_tensorboard:
	docker exec -i -t --name $(CONTAINER_NAME)  $(CMD_TENSORBOARD)

stop:
	docker stop $(CONTAINER_NAME)

rm:	stop
	docker rm $(CONTAINER_NAME)

