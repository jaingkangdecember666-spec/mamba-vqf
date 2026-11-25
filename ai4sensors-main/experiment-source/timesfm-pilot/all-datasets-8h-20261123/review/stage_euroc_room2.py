"""Recover original room-2 sensor CSVs from the published nested ZIP while keeping candidate scoring sealed."""
import json
import struct
import zipfile
import zlib
from pathlib import Path
from urllib.request import Request,urlopen

from stage_supplemental import OUT,copy_checked,persist
from verify_euroc_acquisition import RangeReader

ORIGIN='https://www.research-collection.ethz.ch/entities/researchdata/bcaf173e-5dac-484b-bc37-faf97a594f1f'
OFFICIAL='https://www.research-collection.ethz.ch/server/api/core/bitstreams/ea12bc01-3677-4b4c-853d-87c7870b8c44/content'
URL='https://huggingface.co/datasets/GlowBond/EuRoC_MAV_Dataset/resolve/29d08aeb1c8d9dc5c30497cc538d9cc61f872bb7/vicon_room2.zip'
SIZE=6_013_384_949
OUTER_SHA='6daf2cbc2de9a6bc4e02866c99ed01c29a5c7c164756f06c4f72656192977cfc'


def download_compressed(path,start,size):
    while (path.stat().st_size if path.is_file() else 0)<size:
        have=path.stat().st_size if path.is_file() else 0
        n=min(size-have,32*1024*1024)
        a,b=start+have,start+have+n-1
        for attempt in range(4):
            try:
                req=Request(URL+f'?sensor_archive_bytes={a}-{b}',headers={'Range':f'bytes={a}-{b}',
                            'User-Agent':'IMU-data-source-check','Accept-Encoding':'identity'})
                with urlopen(req,timeout=45) as resp:
                    if resp.status!=206 or resp.headers.get('Content-Range')!=f'bytes {a}-{b}/{SIZE}':
                        raise ValueError('Unmatched server range')
                    data=resp.read(n+1)
                if len(data)!=n: raise OSError('Incomplete archive range')
                with path.open('ab') as dst:dst.write(data)
                break
            except (OSError,ValueError):
                if attempt==3:raise
        if have//(128*1024*1024)!=(have+n)//(128*1024*1024):
            print(path.stem,round(100*(have+n)/size),'percent',flush=True)
    assert path.stat().st_size==size


def extract_record(reader,outer,name):
    dest=OUT/'EuRoC'/name
    dest.mkdir(parents=True,exist_ok=True)
    completed=dest/'source-files.json'
    if completed.exists():return json.loads(completed.read_text(encoding='utf8'))
    entry=outer.getinfo('vicon_room2/'+name+'/'+name+'.zip')
    assert entry.compress_type==zipfile.ZIP_DEFLATED
    reader.seek(entry.header_offset)
    head=reader.read(30)
    assert head[:4]==b'PK\x03\x04'
    fl,ex=struct.unpack_from('<HH',head,26)
    begin=entry.header_offset+30+fl+ex
    compressed=dest/'source-member.deflate'
    inner=dest/'source-member.zip'
    download_compressed(compressed,begin,entry.compress_size)
    decoder=zlib.decompressobj(-15);crc=0
    with compressed.open('rb') as src, inner.open('wb') as dst:
        while chunk:=src.read(2*1024*1024):
            payload=decoder.decompress(chunk);dst.write(payload);crc=zlib.crc32(payload,crc)
        payload=decoder.flush();dst.write(payload);crc=zlib.crc32(payload,crc)
    if not decoder.eof or inner.stat().st_size!=entry.file_size or crc!=entry.CRC:
        raise ValueError('The uncompressed acquisition archive does not match its published ZIP CRC')
    with zipfile.ZipFile(inner) as z:
        docs={}
        for chan in ('imu0','state_groundtruth_estimate0'):
            member,=(i for i in z.infolist() if i.filename.endswith('mav0/'+chan+'/data.csv'))
            docs[chan]=copy_checked(z,member,dest/(chan+'.csv'))
    result={'dataset':'EuRoC','record':name,'source_files':docs,'archive_url':URL,'author_deposit':ORIGIN,
            'author_bitstream':OFFICIAL,'archive_bytes':SIZE,'mirror_lfs_sha256':OUTER_SHA,
            'nested_archive':entry.filename,'nested_archive_crc32':f'{entry.CRC:08x}',
            'transferred_compressed_member_bytes':entry.compress_size}
    persist(completed,result)
    compressed.unlink();inner.unlink()
    return result


def main():
    protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf8'))
    names=protocol['EuRoC']['records']
    manifest=OUT/'source-files.json'
    first=json.loads(manifest.read_text(encoding='utf8'))
    recovery_log=OUT/'source-recovery.json'
    if not recovery_log.exists():
        persist(recovery_log,{'prior_errors':first['errors'],'reason':'The small mirror supplies room 1 only. '
                'The same predeclared room-2 records are available in the original nested room-2 deposit; '
                'only raw IMU and the released state CSV are retained. No metrics have been computed.',
                'original_source':ORIGIN,'byte_mirror':URL})
    reader=RangeReader(URL,SIZE)
    with zipfile.ZipFile(reader) as outer:
        for n in names:
            item=extract_record(reader,outer,n)
            latest=json.loads(manifest.read_text(encoding='utf8'))
            latest['records']=[r for r in latest['records'] if (r['dataset'],r['record'])!=('EuRoC',n)]+[item]
            latest['records'].sort(key=lambda r:(r['dataset'],r['record']))
            latest['errors']=[r for r in latest['errors'] if (r['dataset'],r['record'])!=('EuRoC',n)]
            persist(manifest,latest)
            print('Checked two nested member CRCs and copied byte manifests for',n,flush=True)


if __name__=='__main__':
    main()
