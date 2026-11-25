"""Compare the restored gyro with the six channels in EuRoC's original easy-sequence IMU CSV."""
import hashlib
import io
import json
from http.client import IncompleteRead,RemoteDisconnected
import zipfile
from urllib.error import URLError
from urllib.request import Request,urlopen

import numpy as np
from restore_euroc_inputs import OUT,fetch

MIRROR='pepijn223/euroc-mirror'
ARCHIVE='V1_01_easy.zip'


class RangeReader(io.BufferedIOBase):
    def __init__(self,url,size):
        self.url=url
        self.size=size
        self.position=0
        self.transferred=0

    def tell(self):
        return self.position

    def seek(self,offset,whence=0):
        start=(0,self.position,self.size)[whence]
        if start+offset<0:raise ValueError('Invalid archive offset')
        self.position=start+offset
        return self.position

    def seekable(self):
        return True

    def read(self,n=-1):
        size=self.size-self.position
        n=size if n<0 else min(size,n)
        if n<=0:return b''
        if n>8_000_000 or self.transferred+n>20_000_000:
            raise ValueError('Unexpected size for the central directory or sensor file')
        lo,hi=self.position,self.position+n-1
        headers={'User-Agent':'IMU-sensor-source-audit','Range':f'bytes={lo}-{hi}','Accept-Encoding':'identity'}
        for attempt in range(3):
            try:
                with urlopen(Request(self.url+f'?bytes={lo}-{hi}',headers=headers),timeout=25) as response:
                    if response.status != 206 or response.headers.get('Content-Range')!=f'bytes {lo}-{hi}/{self.size}':
                        raise ValueError('Archive server did not return the requested byte interval')
                    data=response.read(n+1)
                break
            except (URLError,TimeoutError,IncompleteRead,RemoteDisconnected):
                if attempt==2:raise
        if len(data)!=n:raise ValueError('Truncated source interval')
        self.transferred+=n
        self.position+=n
        return data


def main():
    info=json.loads(fetch('https://huggingface.co/api/datasets/'+MIRROR))
    revision=info['sha']
    listing=json.loads(fetch('https://huggingface.co/api/datasets/'+MIRROR+'/tree/'+revision))
    item=next(x for x in listing if x['path']==ARCHIVE)
    url='https://huggingface.co/datasets/'+MIRROR+'/resolve/'+revision+'/'+ARCHIVE
    assert item['size']==1_149_702_102
    reader=RangeReader(url,item['size'])
    with zipfile.ZipFile(reader) as archive:
        chunks={}
        for sensor in ('imu0','state_groundtruth_estimate0'):
            entry,=(x for x in archive.infolist() if x.filename.endswith('mav0/'+sensor+'/data.csv'))
            data=archive.read(entry)
            chunks[sensor]=(data,{'entry':entry.filename,'member_crc32':f'{entry.CRC:08x}',
                                  'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
    raw=np.loadtxt(io.BytesIO(chunks['imu0'][0]),delimiter=',',comments='#',
                   dtype=[('t','<i8'),('v','<f8',(6,))])
    origin_csv=(OUT/'V1_01_easy_original.csv').read_bytes()
    g1=np.loadtxt(io.BytesIO(origin_csv),delimiter=',',comments='#',dtype=[('t','<i8'),('v','<f8',(16,))])
    g2=np.loadtxt(io.BytesIO(chunks['state_groundtruth_estimate0'][0]),delimiter=',',comments='#',
                  dtype=[('t','<i8'),('v','<f8',(16,))])
    np.testing.assert_array_equal(g1,g2)
    with np.load(OUT/'V1_01_easy.npz') as data:
        t=data['utc_ns']
        ix=np.searchsorted(raw['t'],t)
        ix=ix.clip(0,len(raw)-1);previous=np.maximum(0,ix-1)
        take=np.where(np.abs(raw['t'][previous]-t)<=np.abs(raw['t'][ix]-t),previous,ix)
        gyro=raw['v'][take,:3];acc=raw['v'][take,3:]
        gerr=np.abs(gyro-data['gyr_rad_s'])
        aerr=np.abs(acc-data['acc_m_s2'])
        assert gerr.max()<2e-6 and aerr.max()<2e-5
        result={'original_dataset':'https://doi.org/10.3929/ethz-b-000690084','archived_file_mirror':url,
                'mirror_revision':revision,'zip_size':reader.size,'downloaded_bytes':reader.transferred,
                'fields_read':{k:v[1] for k,v in chunks.items()},'original_state_csv_numeric_parity':True,
                'raw_imu_rows':len(raw),'time_matched_training_rows':len(t),
                'max_native_time_difference_ns':int(np.max(np.abs(raw['t'][take]-t))),
                'max_abs_reconstructed_gyro_difference_rad_s':float(gerr.max()),
                'max_abs_hdf5_acc_difference_m_s2':float(aerr.max())}
    (OUT/'original-imu-verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(result)


if __name__=='__main__':
    main()
