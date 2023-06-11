#include <stdio.h>
#include <libaio.h>
#include <sys/stat.h>
#include <fcntl.h>
 
#define error() printf("error [%s : %d]\n", __FILE__, __LINE__)
 
#define BUFF_SIZE 51
#define BUFF_CNT 50
 
/*
 * 异步读取BUFF_CNT次__FILE__文件，每次大小为BUFF_SIZE，并输出
 * 本代码只是aio用法的demo，aio实际运用在网络数据读写等需要异步操作的地方
 *
 * 本demo使用libaio库
 * 也可以通过syscall直接调用io_setup/io_submit/io_getevents/io_destroy系统调用
 *
 * 由于io_submit后，kernel会尝试读数据，只有失败后才会放到work_queue中异步retry，所以本demo读出了源文件的所有数据
 */
int main(int argc, char *argv[])
{
    int err;
    io_context_t ctx;
    struct iocb iocbs[BUFF_CNT];
    struct iocb *iocbps[BUFF_CNT];
    char buff[BUFF_CNT][BUFF_SIZE];
    struct io_event events[BUFF_CNT];
    int fd;
    struct stat stat;
    int i, cnt;
 
    fd = open(__FILE__, O_RDONLY);
    if (fd == -1) {
        error();
        return -1;
    }
 
    err = fstat(fd, &stat);
    if (err == -1) {
        error();
        return -1;
    }
 
    /* kernel会检查(iocb->aio_reserved1 || iocb->aio_reserved2),如果为真返回EINVAL，所以在这里初始化为0 */
    memset(iocbs, 0x00, sizeof(iocbs));
    for (i = 0; i < BUFF_CNT && i * (BUFF_SIZE - 1) < stat.st_size; i++) {
        buff[i][BUFF_SIZE - 1] = 0;
        iocbs[i].data = (void *)i;
        iocbs[i].aio_fildes = fd;
        iocbs[i].aio_lio_opcode = IO_CMD_PREAD;
        iocbs[i].aio_reqprio = 0;
        iocbs[i].u.c.buf = buff + i;
        iocbs[i].u.c.nbytes = BUFF_SIZE - 1;
        iocbs[i].u.c.offset = i * (BUFF_SIZE - 1);
        iocbs[i].u.c.flags = 0;
        iocbps[i] = &iocbs[i];
    }
 
    ctx = 0;
    err = io_setup(BUFF_CNT, &ctx);
    if (err != 0) {
        error();
        return -1;
    }
 
    err = io_submit(ctx, i, (struct iocb **)iocbps);
    if (err != i) {
        error();
        return -1;
    }
 
    printf("submit:%d\n", i);
    cnt = io_getevents(ctx, i/2, i, events, NULL);
    if (cnt < 1) {
        error();
        printf("%d\n", err);
        return -1;
    }
 
    err = io_destroy(ctx);
    if (err != 0) {
        error();
        return -1;
    }
 
    printf("getevents:%d\n", cnt);
    printf("fd:%d\n", events[0].obj->aio_fildes);
    for (i = 0; i < cnt; i++) {
        printf("%s", buff + (int)events[i].data);
    }
 
    return 0;
}