// audio + stt/tts pipeline

let npcVoiceEnabled=true;
let npcAudio=null;
let npcAudioUnlocked=false;
let npcAudioContext=null;
let npcActiveAudio=null;
let npcActiveSpeaker='';
let npcActiveEndPromise=Promise.resolve();
const ttsPendingMap=new Map();
const ttsResolvedCache=new Map();
const MAX_TTS_CACHE_ITEMS=240;
async function unlockNpcAudio(){
    if(npcAudioUnlocked)return true;
    try{
        const AC=window.AudioContext||window.webkitAudioContext;
        if(AC){
            if(!npcAudioContext)npcAudioContext=new AC();
            if(npcAudioContext.state==='suspended')await npcAudioContext.resume();
        }
        // Tiny silent wav to satisfy autoplay policy under user gesture.
        const probe=new Audio('data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=');
        probe.muted=true;
        await probe.play();
        probe.pause();
        probe.currentTime=0;
        npcAudioUnlocked=true;
        return true;
    }catch(e){
        console.warn('[TTS] audio unlock blocked:',e);
        return false;
    }
}
function updateNpcVoiceButton(){
    const btn=$('npcVoiceToggle');
    if(!btn)return;
    btn.classList.toggle('on',npcVoiceEnabled);
    btn.textContent=npcVoiceEnabled?'NPC语音: 开':'NPC语音: 关';
}
function stopNpcVoice(){
    if(npcAudio){
        npcAudio.pause();
        npcAudio.currentTime=0;
    }
    if(npcActiveAudio){
        npcActiveAudio.pause();
    }
    npcActiveAudio=null;
    npcActiveSpeaker='';
    npcActiveEndPromise=Promise.resolve();
}
function toggleNpcVoice(){
    npcVoiceEnabled=!npcVoiceEnabled;
    if(!npcVoiceEnabled)stopNpcVoice();
    else unlockNpcAudio();
    updateNpcVoiceButton();
}
function _speakerProfileFromName(speaker){
    const p=(chars||[]).find(c=>(c.n||c.name||'')===speaker)||null;
    if(!p)return null;
    return {
        name:p.n||p.name||'',
        role:p.r||p.role||'',
        personality:p.p||p.personality||'',
        background:p.b||p.background||'',
        identity:p.identity||p.identity_tag||'',
        gender:p.gender||p.sex||'',
        age:p.age||null,
        age_group:p.ageGroup||p.age_group||'',
        tts_role:p.tts_role||p.ttsRole||p['tts角色']||'',
        'tts角色':p['tts角色']||p.tts_role||p.ttsRole||'',
        tts_voice:p.tts_voice||p.voice||''
    };
}
function _ttsKey(text,emotion,speaker){
    return `${speaker||''}|${emotion||'neutral'}|${text||''}`;
}
function seedNpcSpeechCache(text,emotion,speaker,url){
    const clean=String(text||'').replace(/^\s*[^：:]{1,6}[：:]\s*/,'').trim();
    const u=String(url||'').trim();
    if(!clean||!u)return;
    const key=_ttsKey(clean,emotion||'neutral',speaker||'');
    ttsResolvedCache.set(key,u);
    if(ttsResolvedCache.size>MAX_TTS_CACHE_ITEMS){
        const oldest=ttsResolvedCache.keys().next().value;
        if(oldest!==undefined)ttsResolvedCache.delete(oldest);
    }
}
async function _requestTTSUrl(text,emotion,speaker,speakerProfile){
    const r=await fetch('/api/tts',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({
        text:text,
        emotion:emotion||'neutral',
        speaker:speaker||'',
        speaker_profile:speakerProfile||null
    })});
    const d=await r.json();
    if(d.success&&d.data&&d.data.url)return d.data.url;
    console.warn('[TTS] API failed:',d);
    return '';
}
async function _getTTSUrl(text,emotion,speaker){
    const key=_ttsKey(text,emotion,speaker);
    if(ttsResolvedCache.has(key))return ttsResolvedCache.get(key);
    if(ttsPendingMap.has(key))return ttsPendingMap.get(key);
    const promise=_requestTTSUrl(text,emotion,speaker,_speakerProfileFromName(speaker))
        .catch((e)=>{console.warn('[TTS] request failed:',e);return '';});
    ttsPendingMap.set(key,promise);
    const url=await promise;
    ttsPendingMap.delete(key);
    if(url){
        ttsResolvedCache.set(key,url);
        if(ttsResolvedCache.size>MAX_TTS_CACHE_ITEMS){
            const oldest=ttsResolvedCache.keys().next().value;
            if(oldest!==undefined)ttsResolvedCache.delete(oldest);
        }
    }
    return url;
}
function prefetchNpcSpeech(text,emotion,speaker){
    const clean=String(text||'').replace(/^\s*[^：:]{1,6}[：:]\s*/,'').trim();
    if(!clean||!npcVoiceEnabled)return;
    const key=_ttsKey(clean,emotion||'neutral',speaker||'');
    if(ttsResolvedCache.has(key)||ttsPendingMap.has(key))return;
    const promise=_requestTTSUrl(clean,emotion||'neutral',speaker||'',_speakerProfileFromName(speaker))
        .catch((e)=>{console.warn('[TTS] prefetch failed:',e);return '';});
    ttsPendingMap.set(key,promise);
    promise.then((url)=>{
        ttsPendingMap.delete(key);
        if(!url)return;
        ttsResolvedCache.set(key,url);
        if(ttsResolvedCache.size>MAX_TTS_CACHE_ITEMS){
            const oldest=ttsResolvedCache.keys().next().value;
            if(oldest!==undefined)ttsResolvedCache.delete(oldest);
        }
    }).catch(()=>{
        ttsPendingMap.delete(key);
    });
}
async function waitForAudioTurn(nextSpeaker,nextDelayMs=700){
    const audio=npcActiveAudio;
    if(!audio||audio.ended||audio.paused)return;
    await npcActiveEndPromise.catch(()=>{});
}
async function _playNpcAudio(url,speaker){
    if(!url||!npcVoiceEnabled)return;
    npcAudio=new Audio(url);
    npcActiveAudio=npcAudio;
    npcActiveSpeaker=speaker||'';
    npcActiveEndPromise=new Promise((resolve)=>{
        npcAudio.onended=()=>resolve();
        npcAudio.onerror=()=>resolve();
        npcAudio.onabort=()=>resolve();
    });
    try{
        await npcAudio.play();
    }catch(e){
        console.warn('[TTS] play blocked/failed:',e);
    }
    await npcActiveEndPromise;
}
async function speakNPC(text,emotion,speaker){
    if(!npcVoiceEnabled)return;
    const clean=String(text||'').replace(/^\s*[^：:]{1,6}[：:]\s*/,'').trim();
    if(!clean)return;
    const url=await _getTTSUrl(clean,emotion||'neutral',speaker||'');
    if(!url)return;
    await _playNpcAudio(url,speaker||'');
}


let micButton=null;
let micStream=null;
let audioContext=null;
let recorderNode=null;
let recordingBuffer=[];
let lastVoiceFeatures=null;
let lastVoiceText='';
let lastVoiceEmotion='neutral';
let recordingStart=0;
let recordSampleRate=44100;

function setMicStatus(text){const el=$('micStatus');if(el)el.textContent=text}
function setVoiceEmotion(text){const el=$('voiceEmotion');if(el)el.textContent=`语音情感: ${text||'--'}`}

function downsampleBuffer(buffer, inRate, outRate){
    if(outRate===inRate)return buffer;
    const sampleRateRatio=inRate/outRate;
    const newLength=Math.round(buffer.length/sampleRateRatio);
    const result=new Float32Array(newLength);
    let offsetResult=0;
    let offsetBuffer=0;
    while(offsetResult<result.length){
        const nextOffsetBuffer=Math.round((offsetResult+1)*sampleRateRatio);
        let accum=0, count=0;
        for(let i=offsetBuffer;i<nextOffsetBuffer&&i<buffer.length;i++){accum+=buffer[i];count++}
        result[offsetResult]=accum/count;
        offsetResult++;
        offsetBuffer=nextOffsetBuffer;
    }
    return result;
}
function encodeWav(samples, sampleRate){
    const buffer=new ArrayBuffer(44+samples.length*2);
    const view=new DataView(buffer);
    const writeString=(offset,str)=>{for(let i=0;i<str.length;i++)view.setUint8(offset+i,str.charCodeAt(i))};
    writeString(0,'RIFF');
    view.setUint32(4,36+samples.length*2,true);
    writeString(8,'WAVE');
    writeString(12,'fmt ');
    view.setUint32(16,16,true);
    view.setUint16(20,1,true);
    view.setUint16(22,1,true);
    view.setUint32(24,sampleRate,true);
    view.setUint32(28,sampleRate*2,true);
    view.setUint16(32,2,true);
    view.setUint16(34,16,true);
    writeString(36,'data');
    view.setUint32(40,samples.length*2,true);
    let offset=44;
    for(let i=0;i<samples.length;i++){
        const s=Math.max(-1,Math.min(1,samples[i]));
        view.setInt16(offset, s<0?s*0x8000:s*0x7FFF, true);
        offset+=2;
    }
    return new Blob([view],{type:'audio/wav'});
}
function startLocalRecording(stream){
    recordingBuffer=[];
    recordingStart=Date.now();
    audioContext=new (window.AudioContext||window.webkitAudioContext)();
    recordSampleRate=audioContext.sampleRate;
    const source=audioContext.createMediaStreamSource(stream);
    recorderNode=audioContext.createScriptProcessor(4096,1,1);
    recorderNode.onaudioprocess=(e)=>{
        const input=e.inputBuffer.getChannelData(0);
        recordingBuffer.push(new Float32Array(input));
    };
    source.connect(recorderNode);
    recorderNode.connect(audioContext.destination);
}
async function stopLocalRecording(){
    if(!audioContext)return null;
    recorderNode.disconnect();
    recorderNode=null;
    await audioContext.close();
    audioContext=null;
    const length=recordingBuffer.reduce((acc,cur)=>acc+cur.length,0);
    const merged=new Float32Array(length);
    let offset=0;
    recordingBuffer.forEach(buf=>{merged.set(buf,offset);offset+=buf.length});
    const downsampled=downsampleBuffer(merged,recordSampleRate,16000);
    return encodeWav(downsampled,16000);
}

async function toggleM2(){
    if(!micButton)micButton=document.getElementById('micInputBtn');
    const b=$('mmb');
    const micId=$('micSelect')?$('micSelect').value:'';
    if(isM){
        isM=0;
        if(micStream)micStream.getTracks().forEach(t=>t.stop());
        if(b)b.textContent='🎤 开始录音';
        if(b)b.classList.remove('on');
        if(micButton)micButton.classList.remove('active');
        setMicStatus('正在识别...');
        const wavBlob=await stopLocalRecording();
        if(wavBlob){
            if((Date.now()-recordingStart)<450){
                setMicStatus('录音过短，请至少说半秒');
                return;
            }
            await submitLocalSTT(wavBlob);
        }else{
            setMicStatus('未采集到音频');
        }
        return;
    }
    try{
        const constraints={audio: micId ? {deviceId:{exact:micId}} : true};
        micStream=await navigator.mediaDevices.getUserMedia(constraints);
        isM=1;
        if(b)b.textContent='⏹️ 停止录音';
        if(b)b.classList.add('on');
        if(micButton)micButton.classList.add('active');
        setMicStatus('录音中...');
        startLocalRecording(micStream);
    }catch(e){
        alert('无法开启麦克风: '+e.message);
    }
}

async function submitLocalSTT(wavBlob){
    try{
        const fd=new FormData();
        fd.append('file', wavBlob, 'speech.wav');
        const r=await fetch('/api/stt',{method:'POST',body:fd});
        const d=await r.json();
        if(d.success){
            const text=d.data.text||'';
            const input=$('ci2');
            if(input)input.value=text;
            lastVoiceText=text;
            lastVoiceFeatures=d.data.voice_features||null;
            const emo=d.data.emotion_state?.primary_emotion||'neutral';
            lastVoiceEmotion=emo;
            setVoiceEmotion(emo);
            setMicStatus('识别完成');
        }else{
            setMicStatus('识别失败: '+(d.error||'未知错误'));
        }
    }catch(e){
        setMicStatus('识别失败: '+(e.message||'网络错误'));
    }
}

