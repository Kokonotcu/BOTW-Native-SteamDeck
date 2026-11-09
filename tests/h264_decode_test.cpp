// Decodes a game .mp4 the way runtime/src/misc/h264.cpp does on Linux (one Annex-B access unit per
// call, one thread, no cropping) and checks the pictures: 4:2:0, in display order (pts never goes
// back), and at most a small reorder delay. The movies are High profile with many B-frames.
// Needs an FFmpeg with the mov demuxer and h264_mp4toannexb (tools/deck/ffmpeg-test.sh).
// usage: h264_decode_test <movie.mp4>...   (BOTW_TEST_LOW_DELAY=1 adds AV_CODEC_FLAG_LOW_DELAY)
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavcodec/bsf.h>
#include <libavformat/avformat.h>
}
#include <cassert>
#include <cstdio>
#include <cstdlib>

static int check(const char* path, bool low_delay) {
    AVFormatContext* fmt = nullptr;
    if (avformat_open_input(&fmt, path, nullptr, nullptr) < 0 || avformat_find_stream_info(fmt, nullptr) < 0) {
        printf("FAIL %s: cannot open\n", path);
        return 1;
    }
    int vi = av_find_best_stream(fmt, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    assert(vi >= 0);
    AVCodecParameters* par = fmt->streams[vi]->codecpar;

    const AVBitStreamFilter* f = av_bsf_get_by_name("h264_mp4toannexb");
    AVBSFContext* bsf = nullptr;
    assert(f && av_bsf_alloc(f, &bsf) == 0);
    avcodec_parameters_copy(bsf->par_in, par);
    assert(av_bsf_init(bsf) == 0);

    const AVCodec* codec = avcodec_find_decoder(AV_CODEC_ID_H264);
    AVCodecContext* av = avcodec_alloc_context3(codec);
    av->thread_count = 1;
    if (low_delay) av->flags |= AV_CODEC_FLAG_LOW_DELAY;
    av->apply_cropping = 0;
    assert(avcodec_open2(av, codec, nullptr) == 0);

    AVPacket* pkt = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();
    int units = 0, pictures = 0, bframes = 0, badfmt = 0, backwards = 0, w = 0, h = 0;
    int64_t last_pts = INT64_MIN;
    while (av_read_frame(fmt, pkt) >= 0) {
        if (pkt->stream_index != vi) {
            av_packet_unref(pkt);
            continue;
        }
        av_bsf_send_packet(bsf, pkt);
        while (av_bsf_receive_packet(bsf, pkt) == 0) {
            units++;
            avcodec_send_packet(av, pkt);
            while (avcodec_receive_frame(av, frame) == 0) {
                pictures++;
                if (frame->pict_type == AV_PICTURE_TYPE_B) bframes++;
                if (frame->format != AV_PIX_FMT_YUV420P && frame->format != AV_PIX_FMT_YUVJ420P) badfmt++;
                if (frame->pts != AV_NOPTS_VALUE) {
                    if (frame->pts < last_pts) backwards++;
                    last_pts = frame->pts;
                }
                w = frame->width, h = frame->height;
                av_frame_unref(frame);
            }
            av_packet_unref(pkt);
        }
    }
    int delay = units - pictures;
    int bad = badfmt || backwards || !pictures || delay < 0 || delay > 4;
    printf("%s %s: %dx%d profile %d, %d units, %d pictures (delay %d), %d B, %d out of order, %d bad format\n",
           bad ? "FAIL" : "OK  ", path, w, h, par->profile, units, pictures, delay, bframes, backwards, badfmt);
    av_frame_free(&frame);
    av_packet_free(&pkt);
    avcodec_free_context(&av);
    av_bsf_free(&bsf);
    avformat_close_input(&fmt);
    return bad;
}

int main(int argc, char** argv) {
    const char* e = getenv("BOTW_TEST_LOW_DELAY");
    bool low_delay = e && e[0] == '1';
    int bad = 0;
    for (int i = 1; i < argc; i++) bad += check(argv[i], low_delay);
    printf("%s: %d of %d movies failed\n", low_delay ? "with LOW_DELAY" : "default", bad, argc - 1);
    return bad ? 1 : 0;
}
