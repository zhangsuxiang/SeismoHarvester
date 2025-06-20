#!/usr/bin/env python
# coding: utf-8
# Author: Suxiang Zhang 
# 2025/06/20

import seisbench
import seisbench.data as sbd
import seisbench.util as sbu

import obspy
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNException
import http.client
from pathlib import Path
from collections import defaultdict
import time
import logging
from datetime import datetime
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue
import pickle
import numpy as np

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scedc_2024_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OptimizedSCEDCDownloader:
    def __init__(self, max_workers=8, batch_size=50, cache_events=True):
        """
        优化的SCEDC下载器
        
        参数:
        - max_workers: 最大并发线程数
        - batch_size: 批处理大小
        - cache_events: 是否缓存事件目录
        """
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.cache_events = cache_events
        
        # 创建多个客户端实例来避免连接瓶颈
        self.clients = [Client("SCEDC", timeout=120) for _ in range(max_workers)]
        self.client_queue = Queue()
        for client in self.clients:
            self.client_queue.put(client)
        
        # 统计信息
        self.stats = {
            'total_events': 0,
            'total_traces': 0,
            'successful_traces': 0,
            'failed_downloads': 0,
            'cache_hits': 0,
            'overlapping_traces': 0
        }
        
        # 缓存
        self.event_cache = {}
        self.waveform_cache = {}
        
    def get_client(self):
        """获取一个可用的客户端"""
        return self.client_queue.get()
    
    def return_client(self, client):
        """归还客户端"""
        self.client_queue.put(client)

    def get_event_params(self, event):
        """获取地震事件参数（与原版相同）"""
        origin = event.preferred_origin()
        mag = event.preferred_magnitude()
        sid = str(event.resource_id)
        
        params = {
            "source_id": sid,
            "source_origin_time": str(origin.time),
            "source_origin_uncertainty_sec": origin.time_errors.get("uncertainty") or 0,
            "source_latitude_deg": origin.latitude,
            "source_latitude_uncertainty_km": origin.latitude_errors.get("uncertainty") or 0,
            "source_longitude_deg": origin.longitude,
            "source_longitude_uncertainty_km": origin.longitude_errors.get("uncertainty") or 0,
            "source_depth_km": origin.depth / 1e3 if origin.depth else 0,
            "source_depth_uncertainty_km": ((origin.depth_errors.get("uncertainty") or 0) / 1e3) if origin.depth_errors else 0,
        }
        
        if mag is not None:
            params.update({
                "source_magnitude": mag.mag,
                "source_magnitude_uncertainty": mag.mag_errors.get("uncertainty") or 0,
                "source_magnitude_type": mag.magnitude_type or "",
                "source_magnitude_author": (mag.creation_info.agency_id if mag.creation_info else "") or "",
                "split": "train" if str(origin.time) < "2024-02-01" else "test"
            })
        
        return params

    def get_trace_params(self, pick):
        """获取波形记录参数"""
        return {
            "station_network_code": pick.waveform_id.network_code or "",
            "station_code": pick.waveform_id.station_code or "",
            "trace_channel": pick.waveform_id.channel_code[:2] if pick.waveform_id.channel_code else "",
            "station_location_code": pick.waveform_id.location_code or "",
        }

    def get_events_cached(self, t0, t1, client):
        """缓存版本的事件获取"""
        cache_key = f"{t0}_{t1}"
        
        if self.cache_events and cache_key in self.event_cache:
            logger.info(f"使用缓存的事件目录: {t0.date} → {t1.date}")
            self.stats['cache_hits'] += 1
            return self.event_cache[cache_key]
        
        # 获取新的事件目录
        for attempt in range(1, 4):  # 减少重试次数
            try:
                catalog = client.get_events(
                    t0, t1,
                    minmagnitude=1.0,
                    includearrivals=True
                )
                
                if self.cache_events:
                    self.event_cache[cache_key] = catalog
                
                return catalog
                
            except Exception as e:
                logger.warning(f"获取事件第 {attempt} 次失败: {str(e)[:100]}")
                if attempt < 3:
                    time.sleep(min(5 * attempt, 15))  # 减少等待时间
        
        return obspy.core.event.Catalog()

    def process_overlapping_traces(self, stream):
        """
        处理重叠的波形记录
        返回处理后的stream，避免重叠
        """
        if len(stream) <= 1:
            return stream
        
        # 按照台站、通道、位置码分组
        trace_groups = defaultdict(list)
        for tr in stream:
            key = (tr.stats.network, tr.stats.station, 
                   tr.stats.location, tr.stats.channel)
            trace_groups[key].append(tr)
        
        # 创建新的stream
        processed_stream = obspy.Stream()
        
        for key, traces in trace_groups.items():
            if len(traces) == 1:
                processed_stream += traces[0]
            else:
                # 多个重叠的traces，需要处理
                # 按开始时间排序
                traces.sort(key=lambda x: x.stats.starttime)
                
                # 检查是否真的有重叠
                has_overlap = False
                for i in range(len(traces) - 1):
                    if traces[i].stats.endtime > traces[i+1].stats.starttime:
                        has_overlap = True
                        break
                
                if not has_overlap:
                    # 没有实际重叠，全部保留
                    for tr in traces:
                        processed_stream += tr
                else:
                    # 有重叠，尝试合并
                    try:
                        # 首先尝试merge
                        temp_stream = obspy.Stream(traces)
                        temp_stream.merge(method=1, fill_value='interpolate')
                        if len(temp_stream) > 0:
                            processed_stream += temp_stream[0]
                            self.stats['overlapping_traces'] += 1
                    except Exception as e:
                        # 合并失败，选择质量最好的trace（通常是最长的）
                        best_trace = max(traces, key=lambda x: len(x.data))
                        processed_stream += best_trace
                        logger.debug(f"合并失败，选择最长的trace: {key}")
                        self.stats['overlapping_traces'] += 1
        
        return processed_stream

    def download_waveform_batch(self, waveform_requests):
        """批量下载波形数据"""
        client = self.get_client()
        results = []
        
        try:
            for req in waveform_requests:
                try:
                    net, sta, loc, ch_pref, start_wf, end_wf = req['params']
                    
                    # 检查缓存
                    cache_key = f"{net}_{sta}_{loc}_{ch_pref}_{start_wf}_{end_wf}"
                    if cache_key in self.waveform_cache:
                        results.append({
                            'success': True,
                            'stream': self.waveform_cache[cache_key],
                            'request': req
                        })
                        continue
                    
                    # 下载波形
                    st = client.get_waveforms(
                        network=net,
                        station=sta,
                        location=loc or "*",
                        channel=f"{ch_pref}*",
                        starttime=start_wf,
                        endtime=end_wf
                    )
                    
                    # 处理重叠的traces
                    st = self.process_overlapping_traces(st)
                    
                    # 简单缓存（限制缓存大小）
                    if len(self.waveform_cache) < 1000:
                        self.waveform_cache[cache_key] = st
                    
                    results.append({
                        'success': True,
                        'stream': st,
                        'request': req
                    })
                    
                except Exception as e:
                    logger.debug(f"下载失败 {req['params'][:4]}: {str(e)[:50]}")
                    results.append({
                        'success': False,
                        'error': str(e),
                        'request': req
                    })
                    self.stats['failed_downloads'] += 1
                    
        finally:
            self.return_client(client)
            
        return results

    def process_event_batch(self, events, writer):
        """批量处理事件"""
        # 收集所有波形请求
        all_waveform_requests = []
        event_data = []
        
        for event in events:
            try:
                evp = self.get_event_params(event)
                
                # 按台站+通道分组
                station_groups = defaultdict(list)
                for pick in event.picks:
                    if not pick.waveform_id.channel_code:
                        continue
                    key = (
                        pick.waveform_id.network_code,
                        pick.waveform_id.station_code,
                        pick.waveform_id.location_code,
                        pick.waveform_id.channel_code[:2]
                    )
                    station_groups[key].append(pick)

                # 为每个台站组创建波形请求
                for (net, sta, loc, ch_pref), picks in station_groups.items():
                    start_wf = min(p.time for p in picks) - 60
                    end_wf = max(p.time for p in picks) + 60
                    
                    all_waveform_requests.append({
                        'params': (net, sta, loc, ch_pref, start_wf, end_wf),
                        'event_params': evp,
                        'picks': picks
                    })
                    
            except Exception as e:
                logger.warning(f"处理事件参数失败: {e}")
                continue
        
        # 并行下载波形
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 将请求分批
            batches = [all_waveform_requests[i:i+self.batch_size] 
                      for i in range(0, len(all_waveform_requests), self.batch_size)]
            
            futures = [executor.submit(self.download_waveform_batch, batch) 
                      for batch in batches]
            
            # 处理结果
            for future in as_completed(futures):
                try:
                    batch_results = future.result()
                    
                    for result in batch_results:
                        if not result['success']:
                            continue
                            
                        req = result['request']
                        st = result['stream']
                        evp = req['event_params']
                        picks = req['picks']
                        
                        if len(st) == 0:
                            continue
                        
                        # 检查采样率
                        sr = st[0].stats.sampling_rate
                        if not all(abs(tr.stats.sampling_rate - sr) < 0.1 for tr in st):
                            continue
                        
                        try:
                            # 转换为数组
                            actual_t0, data_arr, _ = sbu.stream_to_array(
                                st,
                                component_order=writer.data_format["component_order"]
                            )
                            
                            # 处理每个pick
                            for pick in picks:
                                try:
                                    self.stats['total_traces'] += 1
                                    trp = self.get_trace_params(pick)
                                    trp["trace_sampling_rate_hz"] = sr
                                    trp["trace_start_time"] = str(actual_t0)
                                    
                                    sample = int((pick.time - actual_t0) * sr)
                                    phase = pick.phase_hint or "unknown"
                                    trp[f"trace_{phase}_arrival_sample"] = sample
                                    trp[f"trace_{phase}_status"] = pick.evaluation_mode or ""

                                    writer.add_trace({**evp, **trp}, data_arr)
                                    self.stats['successful_traces'] += 1
                                    
                                except Exception as e:
                                    logger.debug(f"处理pick失败: {e}")
                                    continue
                                    
                        except Exception as e:
                            logger.debug(f"数组转换失败: {e}")
                            continue
                            
                except Exception as e:
                    logger.warning(f"批处理结果失败: {e}")
                    continue

    def save_progress(self, current_time):
        """保存进度信息"""
        progress_file = Path("download_progress.txt")
        with open(progress_file, 'w') as f:
            f.write(f"Last processed time: {current_time}\n")
            f.write(f"Total events: {self.stats['total_events']}\n")
            f.write(f"Total traces attempted: {self.stats['total_traces']}\n")
            f.write(f"Successful traces: {self.stats['successful_traces']}\n")
            f.write(f"Failed downloads: {self.stats['failed_downloads']}\n")
            f.write(f"Cache hits: {self.stats['cache_hits']}\n")
            f.write(f"Overlapping traces processed: {self.stats['overlapping_traces']}\n")
            if self.stats['total_traces'] > 0:
                f.write(f"Success rate: {self.stats['successful_traces']/self.stats['total_traces']*100:.1f}%\n")

    def load_progress(self):
        """加载上次的进度"""
        progress_file = Path("download_progress.txt")
        if progress_file.exists():
            try:
                with open(progress_file, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        time_str = lines[0].split(": ")[1].strip()
                        return UTCDateTime(time_str)
            except:
                pass
        return None

    def run(self):
        """主下载程序"""
        logger.info("SCEDC 2024年数据下载")
        logger.info(f"配置: 最大并发 {self.max_workers}, 批处理大小 {self.batch_size}")
        
        # 输出文件路径
        base_path = Path("./SCEDC_2024")
        base_path.mkdir(exist_ok=True)
        
        metadata_path = base_path / "metadata-SCEDC-2024.csv"
        waveforms_path = base_path / "waveforms-SCEDC-2024.hdf5"
        
        # 时间范围：2024年全年
        start_time = UTCDateTime(2024, 1, 1)
        end_time = UTCDateTime(2024, 12, 31, 23, 59, 59)
        period = 3 * 24 * 3600  # 3天一个窗口（更小的窗口，更好的并行性）
        
        # 检查是否有之前的进度
        last_processed = self.load_progress()
        if last_processed:
            start_time = last_processed
            logger.info(f"从上次进度继续: {start_time}")
        
        try:
            with sbd.WaveformDataWriter(metadata_path, waveforms_path) as writer:
                writer.data_format = {
                    "dimension_order": "CW",
                    "component_order": "ZNE",
                    "measurement": "velocity",
                    "unit": "counts",
                    "instrument_response": "not restituted",
                }

                t0 = start_time
                window_count = 0
                
                while t0 < end_time:
                    t1 = min(t0 + period, end_time)
                    window_count += 1
                    logger.info(f"处理时间窗口 {window_count}: {t0.date} → {t1.date}")

                    # 获取事件目录
                    client = self.get_client()
                    try:
                        catalog = self.get_events_cached(t0, t1, client)
                        logger.info(f"获取到 {len(catalog)} 个事件")
                        self.stats['total_events'] += len(catalog)
                    finally:
                        self.return_client(client)

                    if len(catalog) == 0:
                        t0 = t1
                        continue

                    # 将事件分批处理
                    event_batches = [catalog[i:i+self.batch_size] 
                                   for i in range(0, len(catalog), self.batch_size)]
                    
                    for batch_idx, event_batch in enumerate(event_batches):
                        logger.info(f"处理事件批次 {batch_idx + 1}/{len(event_batches)}")
                        self.process_event_batch(event_batch, writer)
                    
                    # 保存进度
                    self.save_progress(t1)
                    
                    # 输出统计信息
                    if self.stats['total_traces'] > 0:
                        success_rate = self.stats['successful_traces'] / self.stats['total_traces'] * 100
                        logger.info(f"当前统计: 事件 {self.stats['total_events']}, "
                                  f"波形 {self.stats['successful_traces']}/{self.stats['total_traces']} "
                                  f"({success_rate:.1f}%), 重叠处理 {self.stats['overlapping_traces']}")
                    
                    # 移动到下一个窗口
                    t0 = t1
                    
                    # 清理缓存以防内存溢出
                    if len(self.waveform_cache) > 2000:
                        self.waveform_cache.clear()
                        logger.info("清理波形缓存")

        except KeyboardInterrupt:
            logger.info("用户中断，保存当前进度...")
            self.save_progress(t0)
            raise
        except Exception as e:
            logger.error(f"程序异常终止: {e}")
            self.save_progress(t0)
            raise

        # 最终统计
        logger.info("=" * 50)
        logger.info("下载完成！")
        logger.info(f"总事件数: {self.stats['total_events']}")
        logger.info(f"成功波形: {self.stats['successful_traces']}")
        logger.info(f"尝试波形: {self.stats['total_traces']}")
        logger.info(f"失败下载: {self.stats['failed_downloads']}")
        logger.info(f"缓存命中: {self.stats['cache_hits']}")
        logger.info(f"处理的重叠波形: {self.stats['overlapping_traces']}")
        if self.stats['total_traces'] > 0:
            logger.info(f"成功率: {self.stats['successful_traces']/self.stats['total_traces']*100:.1f}%")
        logger.info(f"数据保存在: {base_path.absolute()}")
        logger.info("=" * 50)


def main():
    """主函数"""
    # 可调整的参数
    MAX_WORKERS = 6      # 并发线程数，根据网络和CPU调整
    BATCH_SIZE = 30      # 批处理大小
    CACHE_EVENTS = True  # 是否缓存事件目录
    
    downloader = OptimizedSCEDCDownloader(
        max_workers=MAX_WORKERS,
        batch_size=BATCH_SIZE,
        cache_events=CACHE_EVENTS
    )
    
    downloader.run()


if __name__ == "__main__":
    main()