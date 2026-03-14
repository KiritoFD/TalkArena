let sid=null,scene='家庭饭桌试炼',mems=[],chars=[],hist=[],cam=null,mic=null,isC=0,isM=0;
let selectedScenarioId='shandong_dinner';
let selectedPressureTags=[];
let pressureValue=5;
let emotionData={confidence:50,calm:50,nervous:20,focus:50};
let emotionInterval=null;
let talkingHeadTimer=null,lastVoiceLevel=0,lastSpeaker='';
const npcRenderState={};
let isFirstCameraClick=true;
let isFirstMicClick=true;
let isNPCSpeaking=false;
let pendingUtterances=[];
let shouldAwaitUser=true;
let profileSceneActive=null;
let lastPageBeforeProfile='p1';
let isEndingSession=false;
let pendingEndTimeout=null;
let endLoadingTimer=null;
let endLoadingProgress=0;
let lastRenderedReportData=null;
const pool={
"家庭饭桌试炼":{id:"shandong_dinner",icon:"🍜",members:[
{a:"👴",n:"大舅",r:"主陪·长辈",b:"看重礼数与体面，善于在热闹中施压",gender:"male",age_group:"elder",identity:"senior",tts_role:"charles"},
{a:"👵",n:"大妗子",r:"观察者",b:"温和追问细节，擅长把话题落到现实",gender:"female",age_group:"adult",identity:"advisor",tts_role:"claire"},
{a:"🧑",n:"表哥",r:"气氛组",b:"会替长辈推进节奏，也会给你台阶",gender:"male",age_group:"youth",identity:"junior",tts_role:"david"}
]},
"商务饭局谈判":{id:"business_dinner",icon:"🤝",members:[
{a:"👨‍💼",n:"王总",r:"甲方负责人",b:"注重结果与执行，关心合作确定性",gender:"male",age_group:"adult",identity:"senior",tts_role:"charles"},
{a:"👔",n:"李总",r:"乙方商务",b:"善于铺垫关系，强调互利与长期合作",gender:"female",age_group:"adult",identity:"business",tts_role:"diana"},
{a:"🧠",n:"周顾问",r:"风险顾问",b:"盯条款边界和落地风险，追问很尖锐",gender:"male",age_group:"adult",identity:"advisor",tts_role:"benjamin"}
]},
"群面竞争场":{id:"interview",icon:"💼",members:[
{a:"👩‍💼",n:"竞争者A",r:"面试对手",b:"自信强势，善于表现自己",gender:"female",age_group:"youth",identity:"junior",tts_role:"bella"},
{a:"🧑‍💼",n:"竞争者B",r:"面试对手",b:"沉稳细致，回答有条理",gender:"male",age_group:"youth",identity:"junior",tts_role:"david"},
{a:"👨‍💼",n:"竞争者C",r:"面试对手",b:"思维活跃，常有创新观点",gender:"male",age_group:"youth",identity:"junior",tts_role:"david"}
]},
"立场攻防辩论":{id:"debate",icon:"⚔️",members:[
{a:"🟦",n:"正方辩手",r:"主张方",b:"强调收益、效率与可行性",gender:"male",age_group:"adult",identity:"specialist",tts_role:"alex"},
{a:"🟥",n:"反方辩手",r:"质疑方",b:"强调代价、风险与外部性",gender:"male",age_group:"adult",identity:"specialist",tts_role:"alex"},
{a:"🧑‍⚖️",n:"点评席",r:"评审",b:"专抓逻辑漏洞，追问证据来源",gender:"female",age_group:"adult",identity:"advisor",tts_role:"claire"}
]}
};
const presetSceneDescriptions={
"家庭饭桌试炼":"春节返乡家宴，长辈主导节奏，话题围绕工作进展、婚恋与人情往来。你需要稳住礼貌、边界与表达力度。",
"商务饭局谈判":"合作签约前夜的商务晚宴，重点在信任、利益边界与合作节奏。语言要有分寸，既给面子也守底线。",
"群面竞争场":"多人面试现场，与其他应聘者同台竞争。主面试官在旁观察，你需要在竞争中展现优势，答案需结论先行、证据支撑。",
"立场攻防辩论":"围绕公共议题展开攻防，强调定义清晰、证据质量和反驳针对性。避免空泛口号。"
};

const banquetLevelDescriptions={
"formal":"正式商务宴请：高端酒店包间，精心布置的餐桌，双方高层悉数到场。这是礼仪性资源展示，信任建立的前置仪式。着装正式，举止得体，言谈谨慎而有分量。酒过三巡后才逐渐进入正题，重点在建立关系、展示诚意，为后续合作铺路。",
"informal":"非正式摸底：装修雅致的私房菜餐厅，氛围相对轻松。双方试探，话里有话。看似随意的闲聊中暗藏机锋，每一个话题都可能是在打探底线。不需要过于正式，但要时刻保持警觉，听懂弦外之音，同时巧妙地传递自己的立场。",
"truth":"酒后吐真言：酒过数巡，氛围变得热烈而直接。高压下的情感博弈，测试忠诚度。酒精卸下了部分伪装，话语开始变得尖锐和真实。这是考验彼此信任和底线的时刻，需要在保持清醒的同时，应对各种情感和利益的考验。",
"street":"深夜大排档：霓虹灯闪烁的街头，塑料板凳，冰镇啤酒。卸下伪装，进行最后的利益交换。没有了办公室的繁文缛节，大家都露出了最真实的一面。这是敲定最终细节的时刻，直接、务实、不绕弯子，但也要守住自己的核心利益。"
};

const interviewQuestions={
"互联网":{
"产品":"请分享你最近使用的一个产品，分析它的核心需求、用户痛点和你的改进建议。",
"销售":"假设我们要推出一款新的SaaS产品，目标客户是中小企业，你会如何设计销售策略？",
"市场":"请为我们即将上线的社交产品设计一个冷启动的营销策略，预算50万。",
"分析":"如果给你一份用户行为数据，你会从哪些维度进行分析，来提升产品的用户留存？"
},
"金融":{
"产品":"请设计一款面向年轻人群的理财产品，说明它的核心卖点和风控机制。",
"销售":"如何向高净值客户推荐我们的财富管理服务？请模拟一个销售场景。",
"市场":"请为我们银行的信用卡业务设计一个年度营销方案，目标是提升年轻客群的活跃度。",
"分析":"如果发现某款理财产品的赎回率突然上升，你会如何分析原因并给出建议？"
},
"快消":{
"产品":"请为我们的新品奶茶设计一个产品概念，包括口味、包装和定价策略。",
"销售":"如何在3个月内将一款新零食打进本地的连锁超市渠道？",
"市场":"请为我们的品牌设计一个双11的营销活动，目标是提升销售额30%。",
"分析":"如果发现某款产品在某个区域的销量下滑，你会如何分析原因并给出改进建议？"
},
"咨询":{
"产品":"请分享你做过的一个产品咨询项目，说明你的分析框架和最终成果。",
"销售":"如何向一家传统企业销售数字化转型咨询服务？请说明你的销售流程。",
"市场":"请为一家新成立的咨询公司设计品牌定位和市场推广策略。",
"分析":"如果客户说他们的利润率在下降，你会如何进行分析并给出建议？"
}
};

let selectedBanquetLevel='formal';
let drinkingCapacity=0;
const scenes=Object.keys(pool);
function $(id){return document.getElementById(id)}

const dicebearStylePool=['avataaars','pixel-art','lorelei','notionists'];
const dicebearOptionAllow=new Set(['top','accessories','facialHair','clothing','eyes','eyebrows','mouth','skinColor','hairColor','facialHairColor','accessoriesColor','clothingColor','hatColor']);
const useExternalDicebear=false;

function hashSeed(str='npc'){
    let h=0;
    for(let i=0;i<str.length;i++){h=((h<<5)-h)+str.charCodeAt(i);h|=0}
    return Math.abs(h);
}
function pickStyle(seed){
    const idx=hashSeed(seed)%dicebearStylePool.length;
    return dicebearStylePool[idx];
}
function pickBySeed(seed,list,offset=0){
    if(!Array.isArray(list)||list.length===0)return '';
    return list[(hashSeed(seed)+offset)%list.length];
}
function inferGender(member){
    const raw=String(
        `${member?.gender||member?.sex||''} ${member?.n||member?.name||''} ${member?.r||member?.role||''} ${member?.b||member?.background||''} ${member?.p||member?.personality||''}`
    ).toLowerCase();
    if(/female|woman|girl|lady|女|女生|女性|妈妈|阿姨|姐姐|妹妹|大妗子|婶|嫂/.test(raw))return 'female';
    if(/male|man|boy|gentleman|男|男生|男性|叔|伯|爷|哥哥|弟弟|大舅|表哥|主陪/.test(raw))return 'male';
    return 'unknown';
}
function inferAgeGroup(member){
    const explicit=Number(member?.age);
    if(Number.isFinite(explicit)&&explicit>0){
        if(explicit>=55)return 'senior';
        if(explicit>=35)return 'middle';
        return 'young';
    }
    const raw=String(
        `${member?.n||member?.name||''} ${member?.r||member?.role||''} ${member?.b||member?.background||''} ${member?.p||member?.personality||''}`
    ).toLowerCase();
    if(/爷|奶|伯|叔|姑父|舅|妗|长辈|senior|elder|主陪/.test(raw))return 'senior';
    if(/新人|晚辈|学生|实习|候选|junior|intern|student|candidate|表弟|表妹/.test(raw))return 'young';
    return 'middle';
}
function inferIdentity(member){
    const raw=String(`${member?.r||member?.role||''} ${member?.b||member?.background||''} ${member?.p||member?.personality||''}`).toLowerCase();
    if(/老板|总|领导|主任|长辈|面试官|评委|甲方|boss|leader|manager|director|主陪|长者/.test(raw))return 'senior';
    if(/顾问|法务|风控|财务|老师|consult|advisor|legal|risk|观察者/.test(raw))return 'advisor';
    if(/商务|销售|客户|运营|bd|sales|business|client|operation/.test(raw))return 'business';
    if(/新人|晚辈|候选|实习|学生|junior|candidate|intern|student|气氛组/.test(raw))return 'junior';
    if(/技术|研发|程序|工程师|产品|engineer|developer|tech|product/.test(raw))return 'tech';
    return 'neutral';
}
function getSceneProfileOverrides(member){
    const name=String(member?.n||member?.name||'');
    const role=String(member?.r||member?.role||'');
    if(name.includes('大舅')||role.includes('主陪')){
        return {gender:'male',ageGroup:'senior',identity:'senior',mustache:true,glasses:false,smile:0,hairType:'short'};
    }
    if(name.includes('大妗子')||role.includes('观察者')){
        return {gender:'female',ageGroup:'middle',identity:'advisor',mustache:false,glasses:true,smile:0,hairType:'long'};
    }
    if(name.includes('表哥')||role.includes('气氛组')){
        return {gender:'male',ageGroup:'young',identity:'junior',mustache:false,glasses:false,smile:1,hairType:'short'};
    }
    return null;
}
function buildIdentityTraits(member,seed){
    const stableSeed=seed||member?.n||member?.name||'npc';
    const gender=inferGender(member);
    const identity=inferIdentity(member);
    const base={
        style:'avataaars',
        options:{
            top:pickBySeed(stableSeed,['shortHairShortFlat','shortHairTheCaesar','shortHairFrizzle','longHairStraight2','longHairMiaWallace']),
            accessories:pickBySeed(stableSeed,['none','prescription01','prescription02','round'],3),
            facialHair:'none',
            clothing:pickBySeed(stableSeed,['shirtCrewNeck','blazerShirt','blazerSweater','hoodie'],5),
            eyes:pickBySeed(stableSeed,['default','happy','side','squint'],7),
            eyebrows:pickBySeed(stableSeed,['default','upDown','raisedExcited','raisedExcitedNatural'],11),
            mouth:pickBySeed(stableSeed,['default','smile','serious'],13),
        }
    };
    if(gender==='female'){
        base.options.top=pickBySeed(stableSeed,['longHairStraight2','longHairMiaWallace','longHairBob','longHairCurly']);
        base.options.facialHair='none';
    }else if(gender==='male'){
        base.options.top=pickBySeed(stableSeed,['shortHairShortFlat','shortHairTheCaesar','shortHairDreads02','shortHairShortWaved']);
        base.options.facialHair=pickBySeed(stableSeed,['none','beardLight','moustacheFancy'],17);
    }
    if(identity==='senior'){
        base.style='avataaars';
        base.options.clothing=pickBySeed(stableSeed,['blazerShirt','blazerSweater','shirtCrewNeck'],19);
        base.options.accessories=pickBySeed(stableSeed,['prescription02','prescription01','none'],23);
        base.options.mouth=pickBySeed(stableSeed,['serious','default','smile'],29);
    }else if(identity==='advisor'){
        base.style='avataaars';
        base.options.accessories=pickBySeed(stableSeed,['prescription01','prescription02','round'],31);
        base.options.clothing=pickBySeed(stableSeed,['shirtCrewNeck','blazerShirt','shirtScoopNeck'],37);
    }else if(identity==='business'){
        base.style='avataaars';
        base.options.clothing=pickBySeed(stableSeed,['blazerShirt','shirtVNeck','shirtCrewNeck'],41);
        base.options.eyebrows=pickBySeed(stableSeed,['default','raisedExcited','upDown'],43);
    }else if(identity==='junior'){
        base.style='notionists';
        base.options.clothing=pickBySeed(stableSeed,['hoodie','graphicShirt','shirtCrewNeck'],47);
        base.options.mouth=pickBySeed(stableSeed,['smile','default','twinkle'],53);
    }else if(identity==='tech'){
        base.style='pixel-art';
        base.options.clothing=pickBySeed(stableSeed,['hoodie','graphicShirt','shirtCrewNeck'],59);
        base.options.accessories=pickBySeed(stableSeed,['none','round','prescription01'],61);
    }else{
        base.style=pickStyle(stableSeed);
    }
    return base;
}
function normalizeVisualTraits(traits,seed,member){
    const normalized=buildIdentityTraits(member,seed);
    if(traits&&typeof traits==='object'){
        if(traits.style)normalized.style=String(traits.style);
        const opts=traits.options||traits.params||{};
        Object.keys(opts||{}).forEach(k=>{
            if(!dicebearOptionAllow.has(k))return;
            const v=opts[k];
            if(v!==null&&v!==undefined&&String(v).trim()!==''){
                normalized.options[k]=String(v);
            }
        });
    }
    return normalized;
}
function buildDicebearUrl(traits,seed,member){
    const safeSeed=seed||'npc';
    const normalized=normalizeVisualTraits(traits,safeSeed,member);
    const params=new URLSearchParams();
    params.set('seed',safeSeed);
    Object.entries(normalized.options).forEach(([k,v])=>params.set(k,v));
    return `https://api.dicebear.com/7.x/${normalized.style}/svg?${params.toString()}`;
}
function resolveAvatarUrl(member){
    if(!member)return null;
    if(member.avatarUrl)return member.avatarUrl;
    const raw=member.avatar||member.a;
    if(raw&&/^https?:\/\//.test(raw))return raw;
    if(useExternalDicebear){
        const traits=member.visualTraits||member.visual_traits||null;
        return buildDicebearUrl(traits,member.n||member.name||'npc',member);
    }
    return buildFallbackAvatarDataUrl(member);
}
function getAvatarInitials(name){
    const text=String(name||'NPC').trim();
    if(!text)return 'NP';
    if(/[\u4e00-\u9fa5]/.test(text)){
        return text.slice(0,2);
    }
    const parts=text.split(/\s+/).filter(Boolean);
    if(parts.length>=2){
        return (parts[0][0]+parts[1][0]).toUpperCase();
    }
    return text.slice(0,2).toUpperCase();
}
function buildFallbackAvatarDataUrl(member){
    const seed=(member?.n||member?.name||'npc');
    const profileOverride=getSceneProfileOverrides(member)||{};
    const identity=profileOverride.identity||inferIdentity(member);
    const gender=profileOverride.gender||inferGender(member);
    const ageGroup=profileOverride.ageGroup||inferAgeGroup(member);
    const h=hashSeed(`${seed}|${identity}|${gender}|${ageGroup}`);
    const bgPalette=[
        ['#dbeafe','#bfdbfe'],
        ['#dcfce7','#bbf7d0'],
        ['#fef3c7','#fde68a'],
        ['#f3e8ff','#e9d5ff'],
        ['#ffe4e6','#fecdd3']
    ];
    const skinPalette=['#F4C7A1','#EAB38F','#D79A74','#C68662'];
    const hairMale=['#1f2937','#374151','#111827','#4b5563'];
    const hairFemale=['#111827','#3f3f46','#78350f','#4c1d95'];
    const clothByIdentity={
        senior:['#334155','#0f172a'],
        advisor:['#1d4ed8','#2563eb'],
        business:['#0f766e','#115e59'],
        junior:['#7c3aed','#6d28d9'],
        tech:['#0f766e','#0f172a'],
        neutral:['#475569','#334155']
    };
    const bg=bgPalette[h%bgPalette.length];
    const skin=skinPalette[(h+1)%skinPalette.length];
    const hair=(gender==='female'?hairFemale:hairMale)[(h+2)%4];
    const cloth=(clothByIdentity[identity]||clothByIdentity.neutral)[(h+3)%2];
    const eyeColor=['#111827','#1f2937','#0f172a'][(h+4)%3];
    const useGlasses=(profileOverride.glasses===true)||((profileOverride.glasses!==false)&&(identity==='advisor'||identity==='senior')&&(h%2===0));
    const hasBeard=(profileOverride.mustache===true)||((profileOverride.mustache!==false)&&(gender==='male')&&(ageGroup!=='young')&&(h%3===0));
    const smileLevel=profileOverride.smile===1?1:(profileOverride.smile===0?0:((identity==='business'||identity==='junior')?1:0));
    const ageWrinkle=ageGroup==='senior';
    const preferLong=profileOverride.hairType==='long' || (profileOverride.hairType!=='short' && gender==='female');
    const hairTopPath = preferLong
        ? "M18,34 C20,14 34,8 48,8 C64,8 78,16 80,34 L80,42 C74,34 66,30 48,30 C32,30 24,34 18,42 Z"
        : "M20,35 C22,20 34,12 48,12 C62,12 74,20 76,35 L76,40 C69,34 60,32 48,32 C36,32 27,34 20,40 Z";
    const hairSideFemale = preferLong
        ? "<path d='M18,40 C16,52 18,66 24,76 L30,76 C24,64 24,52 26,42 Z' fill='"+hair+"' opacity='0.95'/><path d='M78,40 C80,52 78,66 72,76 L66,76 C72,64 72,52 70,42 Z' fill='"+hair+"' opacity='0.95'/>"
        : "";
    const mouthPath = smileLevel
        ? "M40,62 C44,66 52,66 56,62"
        : "M40,64 C44,62 52,62 56,64";
    const glassesSvg = useGlasses
        ? "<rect x='33' y='48' width='11' height='8' rx='2' fill='none' stroke='#334155' stroke-width='1.5'/><rect x='52' y='48' width='11' height='8' rx='2' fill='none' stroke='#334155' stroke-width='1.5'/><line x1='44' y1='52' x2='52' y2='52' stroke='#334155' stroke-width='1.2'/>"
        : "";
    const beardSvg = hasBeard
        ? "<path d='M37,66 C40,74 56,74 59,66 C57,71 39,71 37,66 Z' fill='#374151' opacity='0.85'/>"
        : "";
    const wrinkleSvg = ageWrinkle
        ? "<line x1='36' y1='46' x2='43' y2='46' stroke='#b08968' stroke-width='0.8' opacity='0.6'/><line x1='53' y1='46' x2='60' y2='46' stroke='#b08968' stroke-width='0.8' opacity='0.6'/>"
        : "";
    const svg=`<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 96 96'>
<defs><linearGradient id='bg' x1='0' y1='0' x2='1' y2='1'><stop offset='0%' stop-color='${bg[0]}'/><stop offset='100%' stop-color='${bg[1]}'/></linearGradient></defs>
<rect x='2' y='2' width='92' height='92' rx='20' fill='url(#bg)'/>
<ellipse cx='48' cy='94' rx='38' ry='20' fill='${cloth}' opacity='0.18'/>
<path d='M18,96 C20,78 30,72 48,72 C66,72 76,78 78,96 Z' fill='${cloth}'/>
<rect x='43' y='66' width='10' height='9' rx='4' fill='${skin}'/>
<circle cx='27' cy='51' r='4' fill='${skin}'/>
<circle cx='69' cy='51' r='4' fill='${skin}'/>
<ellipse cx='48' cy='50' rx='22' ry='24' fill='${skin}'/>
<path d='${hairTopPath}' fill='${hair}'/>
${hairSideFemale}
<circle cx='40' cy='52' r='2.1' fill='${eyeColor}'/>
<circle cx='56' cy='52' r='2.1' fill='${eyeColor}'/>
<path d='M36,47 C38,46 42,46 44,47' stroke='#374151' stroke-width='1.4' fill='none' stroke-linecap='round'/>
<path d='M52,47 C54,46 58,46 60,47' stroke='#374151' stroke-width='1.4' fill='none' stroke-linecap='round'/>
<path d='${mouthPath}' stroke='#7f1d1d' stroke-width='1.5' fill='none' stroke-linecap='round'/>
${glassesSvg}
${beardSvg}
${wrinkleSvg}
</svg>`;
    return `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`;
}
function renderAvatarMarkup(member,wrapperClass){
    const url=resolveAvatarUrl(member);
    if(url){
        return `<div class="${wrapperClass}"><img src="${url}" alt="${member?.n||member?.name||'avatar'}"></div>`;
    }
    const emoji=member?.a||member?.avatar||'🙂';
    return `<div class="${wrapperClass} avatar-emoji">${emoji}</div>`;
}
function renderInlineAvatar(avatar,name){
    if(avatar&&/^(https?:\/\/|data:image\/)/.test(avatar)){
        return `<img src="${avatar}" alt="${name||'avatar'}" style="width:24px;height:24px;border-radius:50%;object-fit:cover;">`;
    }
    return `<span>${avatar||'👤'}</span>`;
}
function ensureMemberVisuals(member){
    if(!member)return member;
    if(!member.visualTraits){
        member.visualTraits=buildIdentityTraits(member,member.n||member.name||'npc');
    }
    if(!member.avatar&&member.a)member.avatar=member.a;
    member.avatarUrl=resolveAvatarUrl(member);
    return member;
}
function mapAICharacter(c){
    const member={
        a:c.avatar||'👤',
        avatar:c.avatar||'👤',
        n:c.name||'NPC',
        r:c.role||'',
        b:c.background||c.personality||'未知',
        p:c.personality||'',
        gender:c.gender||c.sex||'',
        visualTraits:c.visualTraits||c.visual_traits||null
    };
    return ensureMemberVisuals(member);
}

function setDrinkingCapacity(score){
    drinkingCapacity=score;
    const stars=document.querySelectorAll('#drinkingCapacityStars .star');
    stars.forEach((star,index)=>{
        if(index<score){
            star.textContent='★';
            star.classList.add('filled');
        } else {
            star.textContent='☆';
            star.classList.remove('filled');
        }
    });
}

let selectedIndustry='互联网';
let selectedPosition='产品';

function updateInterviewQuestion(){
    if(scene !== '群面竞争场') return;
    
    let question = '';
    const customIndustryVal = $('customIndustryInput').value;
    const customPositionVal = $('customPositionInput').value;
    
    const industry = selectedIndustry === '自定义' ? customIndustryVal : selectedIndustry;
    const position = selectedPosition === '自定义' ? customPositionVal : selectedPosition;
    
    if(interviewQuestions[selectedIndustry] && interviewQuestions[selectedIndustry][selectedPosition]){
        question = interviewQuestions[selectedIndustry][selectedPosition];
    } else if(industry && position){
        question = `请分享一个你在${industry}行业做${position}相关工作的经历，说明你遇到的最大挑战和解决方案。`;
    } else {
        question = '请介绍你自己，并分享一个最能体现你能力的项目经历。';
    }
    
    applySceneInfo(question);
}

async function generateCustomInterviewQuestion(){
    const btn = document.getElementById('aiCustomizeBtn');
    const originalText = btn.textContent;
    
    const customIndustryVal = $('customIndustryInput').value;
    const customPositionVal = $('customPositionInput').value;
    
    const industry = selectedIndustry === '自定义' ? customIndustryVal : selectedIndustry;
    const position = selectedPosition === '自定义' ? customPositionVal : selectedPosition;
    
    if(!industry || !position){
        alert('请先选择或输入行业和岗位');
        return;
    }
    
    try {
        btn.disabled = true;
        btn.textContent = '生成中...';
        
        const r = await fetch('/api/interview/generate_question', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                industry: industry,
                position: position
            })
        });
        
        const d = await r.json();
        if (d.success && d.data && d.data.question) {
            applySceneInfo(d.data.question);
        } else {
            alert('生成失败: ' + (d.error || '未知错误'));
        }
    } catch (e) {
        alert('生成失败: ' + e.message);
    } finally {
        btn.disabled = false;
        btn.textContent = originalText;
    }
}

function selectIndustry(el){
    selectedIndustry=el.dataset.value;
    document.querySelectorAll('#industryTags .interview-tag').forEach(t=>t.classList.remove('selected'));
    el.classList.add('selected');
    
    const customInput=$('customIndustryInput');
    if(selectedIndustry==='自定义'){
        customInput.style.display='block';
        customInput.focus();
    } else {
        customInput.style.display='none';
    }
    
    updateInterviewQuestion();
}

function selectPosition(el){
    selectedPosition=el.dataset.value;
    document.querySelectorAll('#positionTags .interview-tag').forEach(t=>t.classList.remove('selected'));
    el.classList.add('selected');
    
    const customInput=$('customPositionInput');
    if(selectedPosition==='自定义'){
        customInput.style.display='block';
        customInput.focus();
    } else {
        customInput.style.display='none';
    }
    
    updateInterviewQuestion();
}

function isPresetScene(){return !!pool[scene]}
function applySceneInfo(description){
    const sceneDescText=document.getElementById('sceneDescriptionText');
    const sceneDescEdit=document.getElementById('sceneDescriptionEdit');
    sceneDescText.innerText=description||'';
    sceneDescEdit.value=description||'';
    document.getElementById('sceneInfoSection').style.display=description?'block':'none';
    document.getElementById('sceneDescription').style.display=description?'block':'none';
}
function refreshSceneInfoForSelection(){
    const btn=document.getElementById('sceneGenBtn');
    const sceneInfoTitle=document.getElementById('sceneInfoTitle');
    const aiCustomizeBtn=document.getElementById('aiCustomizeBtn');
    const preset=isPresetScene();
    if(preset){
        // 根据场景更新标题和按钮显示
        if(scene === '群面竞争场'){
            sceneInfoTitle.textContent='面试问题';
            aiCustomizeBtn.style.display='inline-block';
        } else {
            sceneInfoTitle.textContent='背景信息';
            aiCustomizeBtn.style.display='none';
        }
        
        // 如果是商务饭局谈判场景，使用酒局等级对应的描述
        if(scene === '商务饭局谈判'){
            applySceneInfo(banquetLevelDescriptions[selectedBanquetLevel]);
            btn.style.display='block';
        } else if(scene === '群面竞争场'){
            updateInterviewQuestion();
            btn.style.display='none';
        } else {
            applySceneInfo(presetSceneDescriptions[scene]||`${scene}场景，角色和背景已预置。`);
            btn.style.display='block';
        }
        document.getElementById('memberSection').style.display='block';
        document.getElementById('mg').style.display='flex';
        document.getElementById('actionButtons').style.display='flex';
        btn.disabled=false;
        btn.textContent='🔄 重新生成场景设定';
        return;
    }
    applySceneInfo('');
    btn.disabled=false;
    btn.textContent='🔄 重新生成场景设定';
    btn.style.display='block';
    sceneInfoTitle.textContent='背景信息';
    aiCustomizeBtn.style.display='none';
}
function detectEmotionLabel(t){
    if(!t)return'neutral';
    if(/[哈哈|高兴|开心|好|不错]/i.test(t))return'happy';
    if(/[尴尬|不好意思|抱歉|难过]/i.test(t))return'sad';
    if(/[不行|不能|不喝|生气|别闹]/i.test(t))return'angry';
    return'neutral';
}
function detectEmotion(t){if(!t)return'😐';if(/[哈哈|高兴|开心|好|不错]/i.test(t))return'😊';if(/[谢谢|感谢|感激]/i.test(t))return'🙏';if(/[尴尬|不好意思|抱歉]/i.test(t))return'😳';if(/[不行|不能|不喝]/i.test(t))return'😤';if(/[干|喝|走一个]/i.test(t))return'🍺';return'😐'}
function buildHeadCard(c){const m=ensureMemberVisuals(c);return `<div class="ci state-idle look-user expr-neutral" data-n="${m.n}"><div class="head">${renderAvatarMarkup(m,'avatar-main')}<span class="avatar-exp">😐</span></div><div class="npc-meta"><div class="cn">${m.n}</div><div class="role">${m.r||''}</div><div class="mood-pill">平静</div></div></div>`}
function setRenderState(name,patch={}){if(!npcRenderState[name])npcRenderState[name]={state:'idle',look:'user',backchannel:''};Object.assign(npcRenderState[name],patch)}
function _resolveExpression(st){
    if(st.state==='speaking')return {key:'engaged',label:'发言中',emoji:'🗣️'};
    if(st.state==='reacting')return {key:'warm',label:'有回应',emoji:'🙂'};
    if(st.state==='listening'&&st.look==='speaker')return {key:'focused',label:'在专注',emoji:'🤔'};
    if(st.state==='listening')return {key:'calm',label:'在聆听',emoji:'😌'};
    return {key:'neutral',label:'平静',emoji:'😐'};
}
function applyRenderState(name){const card=document.querySelector(`.ci[data-n="${name}"]`);if(!card)return;const st=npcRenderState[name]||{state:'idle',look:'user',backchannel:''};card.classList.remove('state-idle','state-listening','state-reacting','state-speaking','look-user','look-speaker','has-backchannel','expr-neutral','expr-calm','expr-focused','expr-engaged','expr-warm');card.classList.add(`state-${st.state}`);card.classList.add(`look-${st.look||'user'}`);const expr=_resolveExpression(st);card.classList.add(`expr-${expr.key}`);const mood=card.querySelector('.mood-pill');if(mood)mood.textContent=expr.label;const exp=card.querySelector('.avatar-exp');if(exp)exp.textContent=expr.emoji}
function blinkRandom(){document.querySelectorAll('#cl .ci').forEach(card=>{if(card.classList.contains('state-speaking'))return;if(Math.random()<0.05){card.classList.add('blink');setTimeout(()=>card.classList.remove('blink'),120)}})}
function renderConversationState(mode,speaker=''){
    const names=chars.map(c=>c.n);
    if(!names.length)return;
    names.forEach((name,idx)=>{
        if(mode==='npc_speaking'){
            const isSpeaker=name===speaker;
            setRenderState(name,{state:isSpeaker?'speaking':'listening',look:isSpeaker?'user':'speaker'});
        }else if(mode==='after_npc'){
            const isSpeaker=name===speaker;
            setRenderState(name,{state:isSpeaker?'reacting':'listening',look:'user'});
        }else if(mode==='user_speaking'){
            setRenderState(name,{state:'listening',look:'user'});
        }else{
            setRenderState(name,{state:'listening',look:'user'});
        }
        applyRenderState(name);
    });
}
function inferBeat(){const confusion=Math.max(0,Math.min(100,(100-emotionData.focus+emotionData.nervous)/2));const stress=Math.max(0,Math.min(100,(emotionData.nervous+(100-emotionData.calm))/2));if(stress>66||confusion>70)return 'controlled_rescue';if(scene.includes('面试')||selectedScenarioId==='interview')return 'pressure_check';return 'table_banter'}
function runNonverbalLoop(){if(talkingHeadTimer)clearInterval(talkingHeadTimer);talkingHeadTimer=setInterval(()=>{if(!$('p3').classList.contains('active'))return;blinkRandom()},1400)}
function goBackFromGame(){
    if(confirm('返回将清除当前对话记录，确定要返回吗？')){
        stopNpcVoice();
        sid=null;
        hist=[];
        pendingUtterances=[];
        if(typeof resetUtteranceBlackboard==='function')resetUtteranceBlackboard([]);
        shouldAwaitUser=true;
        isNPCSpeaking=false;
        lastSpeaker='';
        
        $('mc2').innerHTML='';
        $('cl').innerHTML='';
        $('cb').style.display='none';
        $('interruptBtn').style.display='none';
        updScr(50,50);
        renderConversationState('idle');
        
        show('p2');
    }
}
function logClient(level,message,payload){try{fetch('/api/client-log',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({level,message,payload:payload||{}}),keepalive:true})}catch(e){}}
window.addEventListener('error',e=>{logClient('error','window_error',{message:e.message,filename:e.filename,lineno:e.lineno,colno:e.colno})});
window.addEventListener('unhandledrejection',e=>{logClient('error','unhandled_rejection',{reason:String(e.reason||'unknown')})});
function ensureMonitorPanelVisible(){const panel=$('monitorPanel');if(!panel)return;panel.classList.add('visible')}
function show(p){document.querySelectorAll('.page').forEach(e=>e.classList.remove('active'));const page=$(p);if(!page)return;page.classList.add('active');const loginBtn=document.getElementById('loginBtn');if(loginBtn){if(p==='p3'&&!currentUser){loginBtn.style.display='none'}else if(!currentUser){loginBtn.style.display='block'}}if(p==='p3')ensureMonitorPanelVisible();logClient('info','show_page',{page:p})}
function selectPressureTag(el){
    const tag = el.dataset.tag;
    console.log('点击标签:', tag);
    console.log('当前选中标签:', selectedPressureTags);
    
    // 切换选中状态
    if(el.classList.contains('selected')){
        // 取消选中
        el.classList.remove('selected');
        selectedPressureTags = selectedPressureTags.filter(t => t !== tag);
    } else {
        // 选中
        el.classList.add('selected');
        if(!selectedPressureTags.includes(tag)){
            selectedPressureTags.push(tag);
        }
    }
    
    console.log('更新后选中标签:', selectedPressureTags);
    
    // 处理自定义输入
    const customPressureInput = $('customPressureInput');
    if(selectedPressureTags.includes('自定义')){
        customPressureInput.style.display = 'block';
    } else {
        customPressureInput.style.display = 'none';
    }
    
    // 暂时始终显示压力值滑块，用于调试
    const pressureValueBox = $('pressureValueBox');
    console.log('压力值盒子元素:', pressureValueBox);
    pressureValueBox.style.display = 'flex';
}

function updatePressureValue(value){
    pressureValue = parseInt(value);
    $('pressureDisplay').textContent = value;
}

function selectBanquetLevel(el){
    const level = el.dataset.level;
    selectedBanquetLevel = level;
    
    // 取消所有选中状态
    document.querySelectorAll('.banquet-level-tag').forEach(t=>t.classList.remove('selected'));
    // 选中当前标签
    el.classList.add('selected');
    
    // 更新场景信息
    applySceneInfo(banquetLevelDescriptions[level]);
}

function refreshBanquetLevelInfo(){
    if(scene === '商务饭局谈判'){
        applySceneInfo(banquetLevelDescriptions[selectedBanquetLevel]);
    }
}
function goCfg(){show('p2')}
function selScene(el){
    document.querySelectorAll('.sc').forEach(e=>e.classList.remove('on'));
    el.classList.add('on');
    scene=el.dataset.s;
    const p=pool[scene];
    selectedScenarioId=p?p.id:'shandong_dinner';
    
    // 只在家庭饭桌场景显示压力敏感区
    const pressureSectionWrapper = $('pressureSectionWrapper');
    const customPressureInput = $('customPressureInput');
    const pressureValueBox = $('pressureValueBox');
    
    if(scene.includes('家庭')){
        pressureSectionWrapper.style.display = 'block';
    } else {
        pressureSectionWrapper.style.display = 'none';
        customPressureInput.style.display = 'none';
        pressureValueBox.style.display = 'none';
        // 清除选择
        document.querySelectorAll('.pressure-tag').forEach(t=>t.classList.remove('selected'));
        selectedPressureTags = [];
    }
    
    // 只在商务饭局谈判场景显示酒局等级
    const banquetLevelWrapper = $('banquetLevelWrapper');
    
    if(scene === '商务饭局谈判'){
        banquetLevelWrapper.style.display = 'block';
        // 应用选中的酒局等级对应的场景信息
        applySceneInfo(banquetLevelDescriptions[selectedBanquetLevel]);
    } else {
        banquetLevelWrapper.style.display = 'none';
    }
    
    // 只在群面竞争场场景显示面试信息
    const interviewInfoWrapper = $('interviewInfoWrapper');
    
    if(scene === '群面竞争场'){
        interviewInfoWrapper.style.display = 'block';
    } else {
        interviewInfoWrapper.style.display = 'none';
    }
    
    genMems();
}
function genMems(){
    const p=pool[scene];
    if(p){
        mems=p.members.slice(0,3);
        selectedScenarioId=p.id;
        
        // 设置默认用户身份，根据场景调整
        let userRole = '参与者';
        let userBackground = '作为饭局的参与者，你需要在山东酒桌文化的氛围中得体应对各种情况，展示你的情商和社交能力。';
        
        if(scene.includes('家庭')){
            userRole = '晚辈';
            userBackground = '作为家中的晚辈，你需要在长辈面前展现礼貌和尊重，同时巧妙应对长辈的各种关怀和询问。';
        } else if(scene.includes('商务') || scene.includes('客户')){
            userRole = '部门新人';
            userBackground = '作为公司的新人，你需要在商务宴请中展示专业素养，学会得体应对客户的各种话题和敬酒。';
        } else if(scene.includes('面试')){
            userRole = '候选人';
            userBackground = '你需要结论先行、证据支撑，面对追问保持稳定和可验证性。';
        } else if(scene.includes('辩论')){
            userRole = '辩手';
            userBackground = '你需要定义清晰、证据充分，并对对方核心论点做针对性反驳。';
        }
        
        window.userInfo = {
            a: '👨‍💼',
            n: '你',
            r: userRole,
            b: userBackground
        };
    }else{
        mems=pool['家庭饭桌试炼'].members.slice(0,3);
        selectedScenarioId='shandong_dinner';
        
        // 默认用户信息
        window.userInfo = {
            a: '👨‍💼',
            n: '你',
            r: '参与者',
            b: '作为饭局的参与者，你需要在山东酒桌文化的氛围中得体应对各种情况，展示你的情商和社交能力。'
        };
    }
    renderMems();
    renderScenes();
    refreshSceneInfoForSelection();
}
function renderScenes(){$('sg').innerHTML=scenes.map(s=>{
    const isDebate = s === "立场攻防辩论";
    const displayName = isDebate ? "日常纠纷化解" : s;
    const disabledClass = isDebate ? " disabled" : "";
    const extraStyle = isDebate ? 'style="opacity:0.5;pointer-events:none;cursor:not-allowed;"' : '';
    const extraContent = isDebate ? '<div style="font-size:12px;color:#888;margin-top:4px;">（开发中）</div>' : '';
    return `<div class="sc${s===scene?' on':''}${disabledClass}" data-s="${s}" ${isDebate ? '' : `onclick="selScene(this)"`} ${extraStyle}><div style="font-size:24px">${pool[s].icon}</div><div>${displayName}</div>${extraContent}</div>`;
}).join('')}
function renderMems(){
    // 使用动态用户信息，如果未设置则使用默认值
    const userInfo = ensureMemberVisuals(window.userInfo || {
        a: '👨‍💼',
        n: '你',
        r: '参与者',
        b: '作为饭局的参与者，你需要在山东酒桌文化的氛围中得体应对各种情况，展示你的情商和社交能力。'
    });
    window.userInfo=userInfo;
    
    const userMember = `<div class="mc mc-tooltip" style="border:2px solid #4A90E2;background:#E3F2FD;position:relative;cursor:pointer" data-tooltip="${userInfo.b}">
        <div style="position:absolute;top:-10px;right:-10px;width:60px;height:60px;background:#2196F3;color:#fff;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:bold;transform:rotate(15deg);box-shadow:0 2px 5px rgba(0,0,0,0.2);z-index:10;">你的角色</div>
        <div style="position:absolute;top:5px;right:5px;cursor:pointer;font-size:16px;" onclick="editMember('user')">✏️</div>
        ${renderAvatarMarkup(userInfo,'ma')}
        <div class="mn" style="color:#2196F3;">${userInfo.n}</div>
        <div style="background:#2196F3;color:#fff;padding:4px 8px;border-radius:10px;font-size:12px;margin:5px 0;">${userInfo.r}</div>
        <div style="font-size:13px;color:#666;line-height:1.4;">${userInfo.b.substring(0, 50)}${userInfo.b.length > 50 ? '...' : ''}</div>
    </div>`;
    
    $('mg').innerHTML=mems.map((m,i)=>{
        const member=ensureMemberVisuals(m);
        return `
        <div class="mc mc-tooltip" style="position:relative;cursor:pointer" data-tooltip="${member.b || member.personality || '无详细信息'}">
            <div style="position:absolute;top:5px;right:5px;cursor:pointer;font-size:16px;" onclick="editMember(${i})">✏️</div>
            ${renderAvatarMarkup(member,'ma')}
            <div class="mn">${member.n}</div>
            <div style="background:#E3F2FD;color:#2196F3;padding:4px 8px;border-radius:10px;font-size:12px;margin:5px 0;">${member.r}</div>
            <div style="font-size:13px;color:#666;line-height:1.4;">${(member.b || member.personality || '无详细信息').substring(0, 50)}${(member.b || member.personality || '').length > 50 ? '...' : ''}</div>
        </div>
    `}).join('') + userMember;
}

function toggleSceneEdit() {
    const textDiv = document.getElementById('sceneDescriptionText');
    const editWrapper = document.getElementById('sceneEditWrapper');
    const editArea = document.getElementById('sceneDescriptionEdit');
    
    if (editWrapper.style.display === 'none') {
        // 切换到编辑模式
        editArea.value = textDiv.innerText;
        textDiv.style.display = 'none';
        editWrapper.style.display = 'block';
        editArea.focus();
    } else {
        // 切换回显示模式
        textDiv.innerText = editArea.value;
        textDiv.style.display = 'block';
        editWrapper.style.display = 'none';
    }
}

function confirmSceneEdit() {
    const textDiv = document.getElementById('sceneDescriptionText');
    const editWrapper = document.getElementById('sceneEditWrapper');
    const editArea = document.getElementById('sceneDescriptionEdit');
    
    textDiv.innerText = editArea.value;
    textDiv.style.display = 'block';
    editWrapper.style.display = 'none';
}

async function aiOptimizeContent() {
    const btn = document.getElementById('aiOptimizeBtn');
    const originalText = btn.textContent;
    const editArea = document.getElementById('sceneDescriptionEdit');
    const currentContent = editArea.value.trim();
    
    if (!currentContent) {
        alert('请先输入一些内容再进行优化');
        return;
    }
    
    // 确定场景类型
    let sceneType = 'general';
    if (scene === '家庭饭桌试炼') {
        sceneType = 'family';
    } else if (scene === '商务饭局谈判') {
        sceneType = 'business';
    } else if (scene === '群面竞争场') {
        sceneType = 'interview';
    }
    
    try {
        btn.disabled = true;
        btn.textContent = '优化中...';
        
        const r = await fetch('/api/content/optimize', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                content: currentContent,
                scene_type: sceneType
            })
        });
        
        const d = await r.json();
        if (d.success && d.data && d.data.optimized_content) {
            editArea.value = d.data.optimized_content;
        } else {
            alert('优化失败: ' + (d.error || '未知错误'));
        }
    } catch (e) {
        alert('优化失败: ' + e.message);
    } finally {
        btn.disabled = false;
        btn.textContent = originalText;
    }
}

function editMember(index) {
    let member;
    if (index === 'user') {
        member = window.userInfo || {
            a: '👨‍💼',
            n: '你',
            r: '参与者',
            b: '作为饭局的参与者，你需要在山东酒桌文化的氛围中得体应对各种情况，展示你的情商和社交能力。'
        };
    } else {
        member = mems[index];
    }
    
    const modal = document.createElement('div');
    modal.id = 'editModal';
    modal.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.5);display:flex;align-items:center;justify-content:center;z-index:1000;';
    
    modal.innerHTML = `
        <div style="background:white;border-radius:10px;padding:20px;width:90%;max-width:500px;max-height:80vh;overflow-y:auto;">
            <h3 style="margin:0 0 15px 0;color:#333;">编辑成员信息</h3>
            <div style="margin-bottom:15px;">
                <label style="display:block;margin-bottom:5px;font-weight:bold;color:#555;">姓名</label>
                <input type="text" id="editName" value="${member.n}" style="width:100%;padding:8px;border:1px solid #ddd;border-radius:5px;font-size:14px;">
            </div>
            <div style="margin-bottom:15px;">
                <label style="display:block;margin-bottom:5px;font-weight:bold;color:#555;">角色</label>
                <input type="text" id="editRole" value="${member.r}" style="width:100%;padding:8px;border:1px solid #ddd;border-radius:5px;font-size:14px;">
            </div>
            <div style="margin-bottom:15px;">
                <label style="display:block;margin-bottom:5px;font-weight:bold;color:#555;">背景故事</label>
                <textarea id="editBackground" style="width:100%;min-height:100px;padding:8px;border:1px solid #ddd;border-radius:5px;font-size:14px;resize:vertical;">${member.b}</textarea>
            </div>
            <div style="display:flex;gap:10px;justify-content:flex-end;">
                <button onclick="closeEditModal()" style="padding:8px 16px;border:1px solid #ddd;background:white;border-radius:5px;cursor:pointer;">取消</button>
                <button onclick="saveMemberEdit(${index})" style="padding:8px 16px;border:none;background:#2196F3;color:white;border-radius:5px;cursor:pointer;">保存</button>
            </div>
        </div>
    `;
    
    document.body.appendChild(modal);
}

function closeEditModal() {
    const modal = document.getElementById('editModal');
    if (modal) {
        modal.remove();
    }
}

function saveMemberEdit(index) {
    const name = document.getElementById('editName').value;
    const role = document.getElementById('editRole').value;
    const background = document.getElementById('editBackground').value;
    
    if (index === 'user') {
        window.userInfo.n = name;
        window.userInfo.r = role;
        window.userInfo.b = background;
    } else {
        mems[index].n = name;
        mems[index].r = role;
        mems[index].b = background;
    }
    
    renderMems();
    closeEditModal();
}

async function randMem() {
    try {
        const b = document.querySelector('button[onclick="randMem()"]');
        const originalText = b.textContent;
        
        // 更改按钮文本为动态加载文案
        const loadingMessages = ['正在重新设计人物...', '正在构建新的人物关系...', '正在生成新角色...', '即将完成...'];
        let currentIndex = 0;
        let intervalId;
        
        // 显示加载文案，显示完后停留在最后一个文案
        intervalId = setInterval(() => {
            if (currentIndex < loadingMessages.length) {
                b.textContent = loadingMessages[currentIndex];
                currentIndex++;
            } else {
                // 已经显示完所有文案，停止定时器并保持在最后一个文案
                clearInterval(intervalId);
            }
        }, 1000);
        
        b.disabled = true;
        
        const r = await fetch('/api/scenario/regenerate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                scene_type: selectedScenarioId, 
                scene_name: scene,
                only_characters: true // 只生成成员信息
            })
        });
        
        clearInterval(intervalId);
        
        const d = await r.json();
        if (d.success) {
            // 更新成员信息
            if (d.data.characters && d.data.characters.length > 0) {
                // 只取前3个作为NPC
                mems = d.data.characters.slice(0, 3).map(mapAICharacter);
                
                // 如果AI提供了用户身份信息，则更新全局用户身份
                if (d.data.user_identity) {
                    window.userInfo = ensureMemberVisuals({
                        a: d.data.user_identity.avatar || '👤',
                        avatar: d.data.user_identity.avatar || '👤',
                        n: d.data.user_identity.name || '你',
                        r: d.data.user_identity.role || '参与者',
                        b: d.data.user_identity.background || d.data.user_identity.personality || '作为饭局的参与者，你需要在山东酒桌文化的氛围中得体应对各种情况，展示你的情商和社交能力。',
                        p: d.data.user_identity.personality || '',
                        gender: d.data.user_identity.gender || d.data.user_identity.sex || '',
                        visualTraits: d.data.user_identity.visualTraits || d.data.user_identity.visual_traits || null
                    });
                } else {
                    // 默认用户信息
                    window.userInfo = ensureMemberVisuals({
                        a: '👨‍💼',
                        n: '你',
                        r: '参与者',
                        b: '作为饭局的参与者，你需要在山东酒桌文化的氛围中得体应对各种情况，展示你的情商和社交能力。'
                    });
                }
                
                renderMems();
            }
        } else {
            alert('生成失败: ' + (d.error || '未知错误'));
        }
    } catch (e) {
        console.error('生成成员时出错:', e);
        const b = document.querySelector('button[onclick="randMem()"]');
        b.textContent = '随机换人';
        alert('生成成员时出错，请稍后再试');
    } finally {
        const b = document.querySelector('button[onclick="randMem()"]');
        b.textContent = '随机换人';
        b.disabled = false;
    }
}

async function regenerateScene() {
    // 显示确认框
    if (!confirm('⚠️ 重新生成将覆盖当前编辑内容，确定继续？')) {
        return;
    }
    
    try {
        const b = document.getElementById('sceneGenBtn');
        const originalText = b.textContent;
        
        // 更改按钮文本为动态加载文案
        const loadingMessages = ['正在重新设计场景...', '正在重构人物关系...', '正在生成新设定...', '即将完成...'];
        let currentIndex = 0;
        let intervalId;
        
        // 显示加载文案，显示完后停留在最后一个文案
        intervalId = setInterval(() => {
            if (currentIndex < loadingMessages.length) {
                b.textContent = loadingMessages[currentIndex];
                currentIndex++;
            } else {
                // 已经显示完所有文案，停止定时器并保持在最后一个文案
                clearInterval(intervalId);
            }
        }, 1000);
        
        b.disabled = true;
        
        const r = await fetch('/api/scenario/regenerate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                scene_type: selectedScenarioId, 
                scene_name: scene,
                banquet_level: scene === '商务饭局谈判' ? selectedBanquetLevel : null
            })
        });
        
        clearInterval(intervalId);
        
        const d = await r.json();
        if (d.success) {
            // 更新场景描述
            if (d.data.description) {
                const sceneDescText = document.getElementById('sceneDescriptionText');
                const sceneDescEdit = document.getElementById('sceneDescriptionEdit');
                sceneDescText.innerText = d.data.description;
                sceneDescEdit.value = d.data.description;
                
                // 显示场景信息部分
                document.getElementById('sceneInfoSection').style.display = 'block';
                document.getElementById('sceneDescription').style.display = 'block';
            }
            
            // 更新成员信息
            if (d.data.characters && d.data.characters.length > 0) {
                // 只取前3个作为NPC
                mems = d.data.characters.slice(0, 3).map(mapAICharacter);
                
                // 如果AI提供了用户身份信息，则更新全局用户身份
                if (d.data.user_identity) {
                        window.userInfo = ensureMemberVisuals({
                            a: d.data.user_identity.avatar || '👤',
                            avatar: d.data.user_identity.avatar || '👤',
                            n: d.data.user_identity.name || '你',
                            r: d.data.user_identity.role || '参与者',
                            b: d.data.user_identity.background || d.data.user_identity.personality || '作为饭局的参与者，你需要在山东酒桌文化的氛围中得体应对各种情况，展示你的情商和社交能力。',
                            p: d.data.user_identity.personality || '',
                            gender: d.data.user_identity.gender || d.data.user_identity.sex || '',
                            visualTraits: d.data.user_identity.visualTraits || d.data.user_identity.visual_traits || null
                        });
                    } else {
                        // 默认用户信息
                        window.userInfo = ensureMemberVisuals({
                            a: '👨‍💼',
                            n: '你',
                            r: '参与者',
                            b: '作为饭局的参与者，你需要在山东酒桌文化的氛围中得体应对各种情况，展示你的情商和社交能力。'
                        });
                    }
                
                renderMems();
                
                // 显示成员信息部分
                document.getElementById('memberSection').style.display = 'block';
                document.getElementById('mg').style.display = 'flex';
                document.getElementById('actionButtons').style.display = 'flex';
                
                // 改变按钮文字为"重新生成背景信息"
                alert('✅ 场景设定已重新生成！');
            }
        } else {
            b.textContent = '🔄 重新生成场景设定';
            alert('生成失败: ' + (d.error || '未知错误'));
        }
    } catch (e) {
        console.error('生成场景时出错:', e);
        const b = document.getElementById('sceneGenBtn');
        b.textContent = '🔄 重新生成场景设定';
        alert('❌ 重新生成场景失败：' + e.message);
    } finally {
        const b = document.getElementById('sceneGenBtn');
        b.disabled = false;
        b.textContent = '🔄 重新生成场景设定';
    }
}
function updScr(u,a){$('us').textContent=Math.round(u);$('as').textContent=Math.round(a)}
async function rescue(){
    if(!sid)return;
    try{
        if(typeof window.__setLlmBusy==='function')window.__setLlmBusy(true,'救场建议生成中');
        setCoachHint('救场建议生成中，请稍候...');
        const r=await fetch('/api/chat/rescue',{
            method:'POST',
            headers:{'Content-Type':'application/json'},
            body:JSON.stringify({session_id:sid})
        });
        const d=await r.json();
        if(d.success&&d.data&&d.data.suggestion){
            $('ci2').value=d.data.suggestion;
            setCoachHint('救场建议已生成，可直接发送或微调后发送。');
        }else{
            setCoachHint('救场建议生成失败，请重试。');
        }
    }catch(e){
        setCoachHint('救场建议生成失败，请检查网络后重试。');
    }finally{
        if(typeof window.__setLlmBusy==='function')window.__setLlmBusy(false);
    }
}

function _escapeHtml(v){
    return String(v??'')
        .replace(/&/g,'&amp;')
        .replace(/</g,'&lt;')
        .replace(/>/g,'&gt;')
        .replace(/"/g,'&quot;')
        .replace(/'/g,'&#39;');
}

function normalizeReportData(data){
    const d=(data&&typeof data==='object')?{...data}:{};
    const rawScores=(d.scores&&typeof d.scores==='object')?{...d.scores}:{};
    const legacyToModern={
        emotional:'friendliness',
        reaction:'logic'
    };
    Object.keys(legacyToModern).forEach((oldKey)=>{
        const newKey=legacyToModern[oldKey];
        if(rawScores[newKey]===undefined&&rawScores[oldKey]!==undefined){
            rawScores[newKey]=rawScores[oldKey];
        }
    });
    const scores={
        oily:Number(rawScores.oily??50),
        friendliness:Number(rawScores.friendliness??50),
        logic:Number(rawScores.logic??50),
        humor:Number(rawScores.humor??50),
        respect:Number(rawScores.respect??50),
        total:Number(rawScores.total??0)
    };
    if(!scores.total||!Number.isFinite(scores.total)){
        scores.total=Math.round((scores.oily+scores.friendliness+scores.logic+scores.humor+scores.respect)/5);
    }
    const npcList=Array.isArray(d.npc_os_list)?d.npc_os_list:
        Array.isArray(d.npc_thoughts)?d.npc_thoughts:
        Array.isArray(d.npc_list)?d.npc_list:[];
    let transcript=Array.isArray(d.transcript)?d.transcript:[];
    if(!transcript.length&&typeof d.history_log==='string'&&d.history_log.trim()){
        transcript=d.history_log.split('\n').map(s=>String(s||'').trim()).filter(Boolean);
    }
    d.scores=scores;
    d.npc_os_list=npcList;
    d.transcript=transcript;
    if(!d.summary)d.summary='暂无综合点评';
    if(!d.suggestion)d.suggestion='建议继续训练，逐步提升表达的结构与节奏。';
    return d;
}

function buildReportHtml(data){
    const safeData=normalizeReportData(data||{});
    const npcList=Array.isArray(safeData.npc_os_list)?safeData.npc_os_list:[];
    const transcript=Array.isArray(safeData.transcript)?safeData.transcript:[];
    const scores=(safeData.scores&&typeof safeData.scores==='object')?safeData.scores:{};
    const medalNames={
        '🥇':'社交达人',
        '🥈':'社交能手',
        '🥉':'社交新手',
        '📘':'饭桌木头人'
    };
    const medalName=medalNames[safeData.medal]||'社交新手';
    return `
<div style="display:grid;grid-template-columns:minmax(280px,360px) minmax(0,1fr);gap:24px;width:100%;">
    <section style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:20px;padding:24px;">
        <div style="font-size:12px;color:#64748b;letter-spacing:.08em;text-transform:uppercase;">局后复盘</div>
        <h1 style="margin:8px 0 0;font-size:30px;line-height:1.2;color:#0f172a;">${safeData.scene_name||'未命名场景'}</h1>
        <div style="margin-top:14px;display:inline-flex;align-items:center;gap:8px;background:#fee2e2;color:#991b1b;padding:8px 14px;border-radius:999px;font-weight:800;">${safeData.medal||'🥉'} ${medalName}</div>
        <div style="margin-top:22px;display:flex;justify-content:center;"><canvas id="radarChart" width="280" height="280"></canvas></div>
        <div style="display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin-top:18px;">
            <div style="background:#fff;border:1px solid #e2e8f0;border-radius:14px;padding:12px;"><div style="font-size:12px;color:#64748b;">圆滑度</div><div style="margin-top:4px;font-size:22px;font-weight:800;color:#0f172a;">${scores.oily??50}</div></div>
            <div style="background:#fff;border:1px solid #e2e8f0;border-radius:14px;padding:12px;"><div style="font-size:12px;color:#64748b;">亲和力</div><div style="margin-top:4px;font-size:22px;font-weight:800;color:#0f172a;">${scores.friendliness??50}</div></div>
            <div style="background:#fff;border:1px solid #e2e8f0;border-radius:14px;padding:12px;"><div style="font-size:12px;color:#64748b;">逻辑性</div><div style="margin-top:4px;font-size:22px;font-weight:800;color:#0f172a;">${scores.logic??50}</div></div>
            <div style="background:#fff;border:1px solid #e2e8f0;border-radius:14px;padding:12px;"><div style="font-size:12px;color:#64748b;">幽默感</div><div style="margin-top:4px;font-size:22px;font-weight:800;color:#0f172a;">${scores.humor??50}</div></div>
            <div style="background:#fff;border:1px solid #e2e8f0;border-radius:14px;padding:12px;"><div style="font-size:12px;color:#64748b;">懂规矩</div><div style="margin-top:4px;font-size:22px;font-weight:800;color:#0f172a;">${scores.respect??50}</div></div>
            <div style="background:#fff3cd;border:1px solid #fde68a;border-radius:14px;padding:12px;"><div style="font-size:12px;color:#92400e;">总分</div><div style="margin-top:4px;font-size:24px;font-weight:900;color:#7c2d12;">${scores.total||0}</div></div>
        </div>
    </section>
    <section style="display:flex;flex-direction:column;gap:16px;min-width:0;">
        <div style="background:#fff;border:1px solid #e2e8f0;border-radius:18px;padding:20px;">
            <h3 style="margin:0 0 10px;font-size:18px;color:#0f172a;">综合点评</h3>
            <p style="margin:0;font-size:14px;line-height:1.8;color:#334155;white-space:pre-wrap;">${safeData.summary||'暂无综合点评'}</p>
        </div>
        <div style="background:#fff;border:1px solid #e2e8f0;border-radius:18px;padding:20px;">
            <h3 style="margin:0 0 10px;font-size:18px;color:#0f172a;">NPC 内心 OS</h3>
            <div style="display:flex;flex-direction:column;gap:12px;">
                ${npcList.length?npcList.map(npc=>`<div style="display:flex;gap:12px;align-items:flex-start;background:#f8fafc;border-radius:14px;padding:12px;"><div style="font-size:22px;line-height:1;">${npc.avatar||'👤'}</div><div><div style="font-size:14px;font-weight:800;color:#0f172a;">${npc.name||'NPC'}</div><div style="margin-top:4px;font-size:13px;line-height:1.6;color:#475569;white-space:pre-wrap;">${npc.content||npc.os||npc.thought||'暂无内心独白'}</div></div></div>`).join(''):'<div style="font-size:13px;color:#64748b;">暂无 NPC 侧反馈</div>'}
            </div>
        </div>
        <div style="background:#fff;border:1px solid #e2e8f0;border-radius:18px;padding:20px;">
            <h3 style="margin:0 0 10px;font-size:18px;color:#0f172a;">下一轮提升点</h3>
            <p style="margin:0;font-size:14px;line-height:1.8;color:#334155;white-space:pre-wrap;">${safeData.suggestion||'继续保持稳定表达，逐步提升场景适配度。'}</p>
        </div>
        <div style="background:#fff;border:1px solid #e2e8f0;border-radius:18px;padding:20px;">
            <h3 style="margin:0 0 10px;font-size:18px;color:#0f172a;">对话全程文本</h3>
            <div style="max-height:320px;overflow:auto;background:#f8fafc;border-radius:12px;padding:12px;border:1px solid #e2e8f0;">
                ${transcript.length?`<pre style="margin:0;white-space:pre-wrap;line-height:1.7;font-size:13px;color:#334155;">${_escapeHtml(transcript.map(x=>String(x)).join('\n'))}</pre>`:'<div style="font-size:13px;color:#64748b;">暂无对话文本</div>'}
            </div>
        </div>
        <div style="display:flex;gap:10px;justify-content:flex-end;">
            <button class="view-btn" onclick="saveCurrentReportImage()">保存到本地</button>
            <button class="view-btn" onclick="shareCurrentReport()">分享报告</button>
            <button class="view-btn" onclick="goCfg()">再练一局</button>
        </div>
    </section>
</div>`;
}

function _buildShareText(data){
    const d=(data&&typeof data==='object')?data:{};
    const scores=(d.scores&&typeof d.scores==='object')?d.scores:{};
    return [
        `【TalkArena 复盘分享】`,
        `场景：${d.scene_name||'未命名场景'}`,
        `总分：${scores.total??'--'}`,
        `总结：${d.summary||'暂无总结'}`,
        `建议：${d.suggestion||'暂无建议'}`
    ].join('\n');
}

function _wrapCanvasText(ctx,text,maxWidth){
    const raw=String(text||'').replace(/\r/g,'').split('\n');
    const lines=[];
    raw.forEach(seg=>{
        let line='';
        for(const ch of seg){
            const next=line+ch;
            if(ctx.measureText(next).width>maxWidth&&line){
                lines.push(line);
                line=ch;
            }else{
                line=next;
            }
        }
        lines.push(line||'');
    });
    return lines;
}

function _reportImageDataUrl(data){
    const d=(data&&typeof data==='object')?data:{};
    const scores=(d.scores&&typeof d.scores==='object')?d.scores:{};
    const npcList=Array.isArray(d.npc_os_list)?d.npc_os_list:[];
    const transcript=Array.isArray(d.transcript)?d.transcript:[];
    const transcriptText=transcript.length?transcript.join('\n'):'暂无对话文本';
    const estimatedTranscriptLines=Math.max(3,Math.ceil(transcriptText.length/40)+transcript.length);
    const canvas=document.createElement('canvas');
    canvas.width=1080;
    canvas.height=Math.max(1840,1840 + (estimatedTranscriptLines-8)*24);
    const ctx=canvas.getContext('2d');
    ctx.fillStyle='#f8fafc';
    ctx.fillRect(0,0,canvas.width,canvas.height);
    ctx.fillStyle='#ffffff';
    ctx.strokeStyle='#e2e8f0';
    ctx.lineWidth=2;
    const cardHeight=canvas.height-120;
    ctx.fillRect(40,40,1000,cardHeight);
    ctx.strokeRect(40,40,1000,cardHeight);

    ctx.fillStyle='#0f172a';
    ctx.font='bold 52px sans-serif';
    ctx.fillText('TalkArena 复盘报告',90,130);
    ctx.font='600 34px sans-serif';
    ctx.fillStyle='#334155';
    ctx.fillText(`场景：${d.scene_name||'未命名场景'}`,90,190);
    ctx.fillText(`总分：${scores.total??'--'}`,90,240);

    // Radar chart
    const cx=255, cy=460, rr=150;
    const dims=[
        {k:'oily',label:'圆滑度'},
        {k:'friendliness',label:'亲和力'},
        {k:'logic',label:'逻辑性'},
        {k:'humor',label:'幽默感'},
        {k:'respect',label:'懂规矩'},
    ];
    ctx.strokeStyle='#dbeafe';
    for(let lv=1;lv<=5;lv++){
        const r=rr*lv/5;
        ctx.beginPath();
        for(let i=0;i<dims.length;i++){
            const a=-Math.PI/2 + i*Math.PI*2/dims.length;
            const x=cx + Math.cos(a)*r;
            const y=cy + Math.sin(a)*r;
            if(i===0)ctx.moveTo(x,y); else ctx.lineTo(x,y);
        }
        ctx.closePath();
        ctx.stroke();
    }
    ctx.strokeStyle='#bfdbfe';
    for(let i=0;i<dims.length;i++){
        const a=-Math.PI/2 + i*Math.PI*2/dims.length;
        ctx.beginPath();
        ctx.moveTo(cx,cy);
        ctx.lineTo(cx + Math.cos(a)*rr, cy + Math.sin(a)*rr);
        ctx.stroke();
    }
    ctx.fillStyle='rgba(59,130,246,.24)';
    ctx.strokeStyle='#2563eb';
    ctx.lineWidth=3;
    ctx.beginPath();
    for(let i=0;i<dims.length;i++){
        const val=Math.max(0,Math.min(100,Number(scores[dims[i].k]??50)));
        const a=-Math.PI/2 + i*Math.PI*2/dims.length;
        const x=cx + Math.cos(a)*rr*(val/100);
        const y=cy + Math.sin(a)*rr*(val/100);
        if(i===0)ctx.moveTo(x,y); else ctx.lineTo(x,y);
    }
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle='#1e3a8a';
    ctx.font='22px sans-serif';
    dims.forEach((d0,i)=>{
        const a=-Math.PI/2 + i*Math.PI*2/dims.length;
        const lx=cx + Math.cos(a)*(rr+34);
        const ly=cy + Math.sin(a)*(rr+34);
        ctx.fillText(d0.label,lx-34,ly+6);
    });

    // score cards
    const cardX=520, cardY=300, cardW=460, cardH=330;
    ctx.fillStyle='#f8fafc';
    ctx.fillRect(cardX,cardY,cardW,cardH);
    ctx.strokeStyle='#e2e8f0';
    ctx.strokeRect(cardX,cardY,cardW,cardH);
    const cards=[
        ['圆滑度',scores.oily??50],['亲和力',scores.friendliness??50],
        ['逻辑性',scores.logic??50],['幽默感',scores.humor??50],
        ['懂规矩',scores.respect??50],['总分',scores.total??0],
    ];
    for(let i=0;i<cards.length;i++){
        const row=Math.floor(i/2), col=i%2;
        const x=cardX+20+col*220, y=cardY+20+row*98;
        ctx.fillStyle=(cards[i][0]==='总分')?'#fef3c7':'#ffffff';
        ctx.strokeStyle='#e2e8f0';
        ctx.fillRect(x,y,200,80);
        ctx.strokeRect(x,y,200,80);
        ctx.fillStyle='#64748b';
        ctx.font='20px sans-serif';
        ctx.fillText(String(cards[i][0]),x+12,y+28);
        ctx.fillStyle='#0f172a';
        ctx.font='bold 30px sans-serif';
        ctx.fillText(String(cards[i][1]),x+12,y+64);
    }

    let y=700;
    const block=(title,content)=>{
        ctx.fillStyle='#111827';
        ctx.font='bold 30px sans-serif';
        ctx.fillText(title,90,y);
        y+=18;
        ctx.strokeStyle='#e5e7eb';
        ctx.beginPath();
        ctx.moveTo(90,y+16);
        ctx.lineTo(990,y+16);
        ctx.stroke();
        y+=56;
        ctx.fillStyle='#334155';
        ctx.font='24px sans-serif';
        const lines=_wrapCanvasText(ctx,content||'暂无',900);
        lines.forEach(line=>{ctx.fillText(line,90,y);y+=38;});
        y+=22;
    };
    block('综合点评',d.summary||'暂无综合点评');
    block('下一轮提升点',d.suggestion||'继续保持稳定表达。');
    if(npcList.length){
        const txt=npcList.map(n=>`${n.name||'NPC'}：${n.content||n.os||n.thought||'暂无反馈'}`).join('\n');
        block('NPC 内心 OS',txt);
    }
    block('对话全程文本',transcriptText);
    ctx.fillStyle='#94a3b8';
    ctx.font='20px sans-serif';
    ctx.fillText(`生成时间：${new Date().toLocaleString()}`,90,canvas.height-70);
    return canvas.toDataURL('image/png');
}

function _ensureQrModal(){
    let modal=document.getElementById('shareQrModal');
    if(modal)return modal;
    modal=document.createElement('div');
    modal.id='shareQrModal';
    modal.style.cssText='position:fixed;inset:0;background:rgba(2,6,23,.55);display:none;align-items:center;justify-content:center;z-index:3000;';
    modal.innerHTML=`
<div style="width:min(92vw,460px);background:#fff;border-radius:16px;padding:16px;box-shadow:0 20px 40px rgba(15,23,42,.25);">
  <div style="display:flex;align-items:center;justify-content:space-between;gap:8px;">
    <div style="font-size:18px;font-weight:900;color:#0f172a;">扫码分享报告</div>
    <button id="shareQrCloseBtn" class="view-btn" style="padding:6px 10px;">关闭</button>
  </div>
  <div id="shareQrBody" style="margin-top:12px;display:flex;flex-direction:column;align-items:center;gap:10px;">
    <div style="font-size:13px;color:#64748b;">生成中...</div>
  </div>
</div>`;
    document.body.appendChild(modal);
    modal.addEventListener('click',e=>{if(e.target===modal)modal.style.display='none';});
    const closeBtn=modal.querySelector('#shareQrCloseBtn');
    if(closeBtn)closeBtn.onclick=()=>{modal.style.display='none';};
    return modal;
}

function _openQrModalLoading(){
    const modal=_ensureQrModal();
    modal.style.display='flex';
    const body=modal.querySelector('#shareQrBody');
    if(body){
        body.innerHTML='<div style="font-size:13px;color:#64748b;">正在生成报告图片与二维码...</div>';
    }
}

function _openQrModalResult(qrSrc,shareUrl){
    const modal=_ensureQrModal();
    modal.style.display='flex';
    const body=modal.querySelector('#shareQrBody');
    if(!body)return;
    body.innerHTML=`
<img src="${qrSrc}" alt="share qr" style="width:300px;height:300px;border:1px solid #e2e8f0;border-radius:12px;background:#fff;" />
<div style="font-size:12px;color:#64748b;text-align:center;word-break:break-all;">${shareUrl}</div>
<div style="display:flex;gap:8px;">
  <button id="copyShareLinkBtn" class="view-btn">复制链接</button>
  <a class="view-btn" href="${shareUrl}" target="_blank" rel="noopener">打开链接</a>
</div>`;
    const copyBtn=modal.querySelector('#copyShareLinkBtn');
    if(copyBtn){
        copyBtn.onclick=async()=>{
            try{await navigator.clipboard.writeText(shareUrl);alert('已复制链接');}
            catch(e){alert(shareUrl);}
        };
    }
}

async function shareCurrentReport(){
    const data=lastRenderedReportData||{};
    const text=_buildShareText(data);
    _openQrModalLoading();
    try{
        const imageData=_reportImageDataUrl(data);
        const resp=await fetch('/api/share/report-image',{
            method:'POST',
            headers:{'Content-Type':'application/json'},
            body:JSON.stringify({image_data:imageData})
        });
        const payload=await resp.json();
        if(!payload.success||!payload.data){
            throw new Error(payload.error||'分享生成失败');
        }
        const shareUrl=String(payload.data.public_url||'').trim();
        if(!shareUrl)throw new Error('缺少公网分享地址，请联系管理员配置 TALKARENA_PUBLIC_BASE_URL');
        const qrSrc=`https://api.qrserver.com/v1/create-qr-code/?size=320x320&data=${encodeURIComponent(shareUrl)}`;
        _openQrModalResult(qrSrc,shareUrl);
        return;
    }catch(e){}
    try{
        await navigator.clipboard.writeText(text);
        alert('二维码生成失败，已复制分享文案');
    }catch(e){
        alert(text);
    }
}

function saveCurrentReportImage(){
    try{
        const data=lastRenderedReportData||{};
        const imageData=_reportImageDataUrl(data);
        const a=document.createElement('a');
        const scene=String(data.scene_name||'TalkArena报告').replace(/[\\/:*?"<>|]/g,'_');
        const ts=new Date().toISOString().replace(/[:.]/g,'-');
        a.href=imageData;
        a.download=`${scene}_${ts}.png`;
        document.body.appendChild(a);
        a.click();
        a.remove();
    }catch(e){
        alert('保存图片失败，请重试');
    }
}
function _renderEndLoading(progress=0,stage='正在生成复盘...'){
    const p=Math.max(0,Math.min(100,Number(progress)||0));
    $('rc').innerHTML=`
<div style="max-width:720px;margin:80px auto;background:#fff;border:1px solid #e2e8f0;border-radius:18px;padding:28px;box-shadow:0 10px 30px rgba(15,23,42,.08);">
    <div style="font-size:22px;font-weight:900;color:#0f172a;">复盘生成中</div>
    <div style="margin-top:8px;font-size:14px;color:#64748b;">${stage}</div>
    <div style="margin-top:20px;height:12px;background:#e5e7eb;border-radius:999px;overflow:hidden;">
        <div style="height:100%;width:${p}%;background:linear-gradient(90deg,#2563eb,#7c3aed);transition:width .25s;"></div>
    </div>
    <div style="margin-top:10px;font-size:13px;color:#475569;">${Math.round(p)}%</div>
</div>`;
}
function _startEndLoading(){
    endLoadingProgress=6;
    _renderEndLoading(endLoadingProgress,'正在整理本局表现...');
    show('p4');
    if(endLoadingTimer){clearInterval(endLoadingTimer)}
    const startAt=Date.now();
    endLoadingTimer=setInterval(()=>{
        const elapsed=(Date.now()-startAt)/1000;
        if(endLoadingProgress<92){
            endLoadingProgress=Math.min(92,endLoadingProgress+Math.max(1,Math.round((95-endLoadingProgress)/10)));
            _renderEndLoading(endLoadingProgress,'正在分析对话细节与策略得分...');
            return;
        }
        endLoadingProgress=Math.min(99,endLoadingProgress+0.2);
        const stage=elapsed>15?'模型正在深度生成复盘，请稍候...':'正在整合报告内容...';
        _renderEndLoading(endLoadingProgress,stage);
    },280);
}
function _finishEndLoading(){
    if(endLoadingTimer){clearInterval(endLoadingTimer);endLoadingTimer=null}
    endLoadingProgress=100;
    _renderEndLoading(endLoadingProgress,'复盘生成完成');
}

function getHistoryKey(){
    const key=authToken||currentUser?.uid||currentUser?.email||'guest';
    return `talkarena_history_${key}`;
}
function loadReports(){
    try{
        const raw=localStorage.getItem(getHistoryKey());
        return raw?JSON.parse(raw):[];
    }catch(e){
        return [];
    }
}
function saveReportToHistory(data){
    if(!data)return;
    const reports=loadReports();
    const record={
        id:Date.now().toString(36),
        timestamp:Date.now(),
        scene_name:data.scene_name||'未命名场景',
        summary:data.summary||'综合点评',
        data:data
    };
    reports.unshift(record);
    localStorage.setItem(getHistoryKey(),JSON.stringify(reports.slice(0,200)));
}
function renderProfile(sceneKey){
    const container=$('profileContent');
    if(!container)return;
    if(!currentUser){
        container.innerHTML=`<div class="report-empty">请先登录后查看个人中心。<div style="margin-top:10px;"><button class="view-btn" onclick="openAuthModal()">去登录</button></div></div>`;
        return;
    }
    const reports=loadReports();
    if(reports.length===0){
        container.innerHTML=`<div class="report-empty">暂无历史对话报告，完成一次练习后会自动保存到这里。</div>`;
        return;
    }
    const groups={};
    reports.forEach(r=>{
        const key=r.scene_name||'未命名场景';
        if(!groups[key])groups[key]=[];
        groups[key].push(r);
    });
    const scenes=Object.keys(groups);
    const active=sceneKey||profileSceneActive||scenes[0];
    profileSceneActive=active;
    const total=reports.length;
    const distHtml=scenes.map(s=>`<div class="dist-row"><span>${s}</span><span>${groups[s].length}</span></div>`).join('');
    const tabsHtml=scenes.map(s=>`<button class="profile-tab ${s===active?'active':''}" data-scene="${s}" onclick="renderProfile(this.dataset.scene)">${s}</button>`).join('');
    const cardsHtml=(groups[active]||[]).map(r=>{
        const date=new Date(r.timestamp);
        const dateLabel=`${date.getFullYear()}-${String(date.getMonth()+1).padStart(2,'0')}-${String(date.getDate()).padStart(2,'0')}`;
        return `
        <div class="report-card">
            <h5>${r.scene_name}</h5>
            <p>${r.summary}</p>
            <div style="font-size:12px;color:#94a3b8;">${dateLabel}</div>
            <button class="view-btn" onclick="openHistoryReport('${r.id}')">查看详细报告</button>
        </div>
        `;
    }).join('');
    container.innerHTML=`
        <div class="profile-stats">
            <div class="profile-stat">
                <h4>总练习次数</h4>
                <div class="stat-val">${total}</div>
            </div>
            <div class="profile-stat">
                <h4>各场景练习次数</h4>
                ${distHtml}
            </div>
            <div class="profile-stat">
                <h4>当前场景</h4>
                <div class="stat-val">${active}</div>
            </div>
        </div>
        <div class="profile-tabs">${tabsHtml}</div>
        <div class="report-grid">${cardsHtml}</div>
    `;
}
function openProfile(){
    lastPageBeforeProfile=document.querySelector('.page.active')?.id||'p1';
    show('p5');
    renderProfile();
}
function goBackFromProfile(){
    show(lastPageBeforeProfile||'p1');
}
function renderEndReport(data,source='runtime'){
    const safeData=normalizeReportData((data&&typeof data==='object')?data:{});
    lastRenderedReportData=safeData;
    try{
        const html=buildReportHtml(safeData);
        logClient('info','render_end_report_start',{source,scene_name:safeData.scene_name||'',score_keys:Object.keys(safeData.scores||{}),npc_os_count:Array.isArray(safeData.npc_os_list)?safeData.npc_os_list.length:0,html_length:html.length});
        $('rc').innerHTML=html;
        show('p4');
        drawRadarChart(safeData.scores||{});
        logClient('info','render_end_report_complete',{source});
    }catch(renderErr){
        logClient('error','render_end_report_failed',{source,error:String(renderErr)});
        $('rc').innerHTML=`<div style="max-width:820px;margin:40px auto;background:#fff;padding:24px;border-radius:16px;box-shadow:0 8px 30px rgba(0,0,0,0.08)"><h2 style="margin-top:0">复盘已生成，但页面渲染失败</h2><p><b>场景：</b>${safeData.scene_name||'本轮会话'}</p><p><b>总结：</b>${safeData.summary||'暂无总结'}</p><p><b>建议：</b>${safeData.suggestion||'暂无建议'}</p><pre style="white-space:pre-wrap;background:#f8fafc;padding:12px;border-radius:12px;font-size:12px;overflow:auto">${JSON.stringify(safeData,null,2)}</pre></div>`;
        show('p4');
    }
}
function openHistoryReport(id){
    const reports=loadReports();
    const record=reports.find(r=>r.id===id);
    if(!record)return;
    logClient('info','open_history_report',{id,found:!!record});
    renderEndReport(record.data,'history');
}
function scheduleEndSession(delayMs=0){
    if(isEndingSession)return;
    if(pendingEndTimeout){clearTimeout(pendingEndTimeout)}
    pendingEndTimeout=setTimeout(()=>{pendingEndTimeout=null;end()},delayMs);
}
async function end(){
    if(!sid||isEndingSession)return;
    isEndingSession=true;
    const activeSid=sid;
    stopNpcVoice();
    try{
        const endBtn=document.querySelector('.eb');
        if(endBtn)endBtn.disabled=true;
        _startEndLoading();
        logClient('info','end_clicked',{sid:activeSid});
        const r=await fetch('/api/session/end',{
            method:'POST',
            headers:{'Content-Type':'application/json'},
            body:JSON.stringify({session_id:activeSid})
        });
        if(!r.ok){
            throw new Error(`结束会话接口异常: HTTP ${r.status}`);
        }
        const d=await r.json();
        if(!d.success){
            throw new Error(d.error||'结束会话失败');
        }
        const data=(d.data&&typeof d.data==='object')?d.data:{};
        logClient('info','end_response_received',{sid:activeSid,scene_name:data.scene_name||'',score_keys:Object.keys(data.scores||{}),npc_os_count:Array.isArray(data.npc_os_list)?data.npc_os_list.length:0});
        _finishEndLoading();
        renderEndReport(data,'api');
        try{
            saveReportToHistory(data);
            logClient('info','end_history_saved',{sid:activeSid});
        }catch(saveErr){
            logClient('error','end_history_save_failed',{sid:activeSid,error:String(saveErr)});
        }
        sid=null;
    }catch(e){
        logClient('error','end_failed',{sid:activeSid,error:String(e)});
        _finishEndLoading();
        $('rc').innerHTML=`
<div style="max-width:820px;margin:40px auto;background:#fff;padding:24px;border-radius:16px;box-shadow:0 8px 30px rgba(0,0,0,0.08)">
  <h2 style="margin-top:0;color:#0f172a;">报告生成失败</h2>
  <p style="color:#475569;line-height:1.7;">当前与模型服务连接不稳定，请重试生成报告。</p>
  <div style="margin-top:12px;padding:10px 12px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;font-size:12px;color:#64748b;white-space:pre-wrap;">${_escapeHtml(String(e||''))}</div>
  <div style="margin-top:16px;display:flex;gap:10px;">
    <button class="view-btn" onclick="end()">重试生成报告</button>
    <button class="view-btn" onclick="goCfg()">返回配置页</button>
  </div>
</div>`;
        show('p4');
    }finally{
        const endBtn=document.querySelector('.eb');
        if(endBtn&&sid)endBtn.disabled=false;
        isEndingSession=false;
    }
}

function drawRadarChart(scores){
    const safeScores=(scores&&typeof scores==='object')?scores:{};
    const canvas=document.getElementById('radarChart');
    if(!canvas)return;
    const ctx=canvas.getContext('2d');
    const centerX=canvas.width/2;
    const centerY=canvas.height/2;
    const radius=Math.min(centerX,centerY)-40;
    
    // 五个维度
    const labels=['Oily','Friendly','Logical','Humor','Respect'];
    const values=[
        safeScores.oily||50,
        safeScores.friendliness||50,
        safeScores.logic||50,
        safeScores.humor||50,
        safeScores.respect||50
    ];
    
    // 清空画布
    ctx.clearRect(0,0,canvas.width,canvas.height);
    
    // 绘制背景网格（5 个同心圆）
    ctx.strokeStyle='#e0e0e0';
    ctx.lineWidth=1;
    for(let i=1;i<=5;i++){
        const r=radius*i/5;
        ctx.beginPath();
        ctx.arc(centerX,centerY,r,0,Math.PI*2);
        ctx.stroke();
    }
    
    // 绘制轴线
    ctx.strokeStyle='#d0d0d0';
    for(let i=0;i<5;i++){
        const angle=(Math.PI*2/5)*i-Math.PI/2;
        const x=centerX+Math.cos(angle)*radius;
        const y=centerY+Math.sin(angle)*radius;
        ctx.beginPath();
        ctx.moveTo(centerX,centerY);
        ctx.lineTo(x,y);
        ctx.stroke();
    }
    
    // 绘制数据多边形
    ctx.strokeStyle='#4a5dca';
    ctx.lineWidth=2;
    ctx.fillStyle='rgba(74,93,202,0.2)';
    ctx.beginPath();
    for(let i=0;i<5;i++){
        const angle=(Math.PI*2/5)*i-Math.PI/2;
        const value=values[i]/100;
        const x=centerX+Math.cos(angle)*radius*value;
        const y=centerY+Math.sin(angle)*radius*value;
        if(i===0){
            ctx.moveTo(x,y);
        }else{
            ctx.lineTo(x,y);
        }
    }
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    
    // 绘制数据点
    ctx.fillStyle='#4a5dca';
    for(let i=0;i<5;i++){
        const angle=(Math.PI*2/5)*i-Math.PI/2;
        const value=values[i]/100;
        const x=centerX+Math.cos(angle)*radius*value;
        const y=centerY+Math.sin(angle)*radius*value;
        ctx.beginPath();
        ctx.arc(x,y,4,0,Math.PI*2);
        ctx.fill();
    }
    
    // 绘制标签
    ctx.fillStyle='#666';
    ctx.font='11px Arial';
    ctx.textAlign='center';
    ctx.textBaseline='middle';
    for(let i=0;i<5;i++){
        const angle=(Math.PI*2/5)*i-Math.PI/2;
        const labelX=centerX+Math.cos(angle)*(radius+20);
        const labelY=centerY+Math.sin(angle)*(radius+20);
        ctx.fillText(labels[i],labelX,labelY);
    }
    
    // 绘制刻度值
    ctx.fillStyle='#999';
    ctx.font='10px Arial';
    for(let i=1;i<=5;i++){
        const x=centerX+10;
        const y=centerY-radius*i/5+3;
        ctx.fillText(i*20,x,y);
    }
}
function toggleCameraPanel(){const panel=$('monitorPanel');if(!panel)return;panel.classList.add('visible');panel.scrollIntoView({behavior:'smooth',block:'nearest',inline:'nearest'});logClient('info','camera_panel_focus',{visible:true});}
function toggleMicPanel(){if(isFirstMicClick){alert('欢迎使用麦克风功能！\n\n请选择您的麦克风设备，然后点击"开启麦克风"按钮。');isFirstMicClick=false;}toggleM2();}
async function toggleC(){const b=$('cmb'),vid=$('camVideo'),ph=$('camPlaceholder'),camId=$('camSelect').value;if(isC){if(cam)cam.getTracks().forEach(t=>t.stop());if(emotionInterval)clearInterval(emotionInterval);isC=0;b.textContent='📷 开启摄像头';b.classList.remove('on');vid.pause();vid.srcObject=null;vid.style.display='none';ph.style.display='flex';ph.textContent='摄像头未开启';$('ei').textContent='❓';$('et').textContent='未检测';emotionData={confidence:50,calm:50,nervous:20,focus:50};updateEmotionDisplay()}else{try{const constraints={video:{width:320,height:240,facingMode:'user'}};if(camId)constraints.deviceId={exact:camId};cam=await navigator.mediaDevices.getUserMedia(constraints);isC=1;b.textContent='✅ 已开启';b.classList.add('on');vid.srcObject=cam;vid.style.display='block';ph.style.display='none';vid.play().then(()=>{emotionInterval=setInterval(()=>{if(!isC)return;const eList=[{i:'😊',t:'开心',c:80,n:10,cal:60,f:70},{i:'😎',t:'自信',c:90,n:5,cal:50,f:80},{i:'😐',t:'平静',c:40,n:10,cal:90,f:50},{i:'😰',t:'紧张',c:30,n:90,cal:20,f:40},{i:'🤔',t:'思考',c:60,n:30,cal:70,f:95},{i:'🙂',t:'放松',c:70,n:5,cal:80,f:60},{i:'😤',t:'坚定',c:85,n:15,cal:40,f:75}];const e=eList[Math.floor(Math.random()*eList.length)];$('ei').textContent=e.i;$('et').textContent=e.t;emotionData={confidence:e.c,nervous:e.n,calm:e.cal,focus:e.f};updateEmotionDisplay();console.log('[Emotion] 实时分析:', emotionData)},1500)}).catch(e=>{console.log('播放失败:',e)})}catch(e){alert('无法开启摄像头: '+e.message)}}}
function updateEmotionDisplay(){$('val-confidence').textContent=emotionData.confidence;$('val-calm').textContent=emotionData.calm;$('val-nervous').textContent=emotionData.nervous;$('val-focus').textContent=emotionData.focus;$('bar-confidence').style.width=emotionData.confidence+'%';$('bar-calm').style.width=emotionData.calm+'%';$('bar-nervous').style.width=emotionData.nervous+'%';$('bar-focus').style.width=emotionData.focus+'%'}
function updateMetrics(scores){console.log('[Metrics] 收到分数:', scores);if(scores){const total=Math.round((scores.emotional_intelligence+scores.response_quality+scores.pressure_handling+scores.cultural_fit)/4);$('val-score').textContent=total;$('bar-score').style.width=total+'%'}else{console.log('[Metrics] 分数为空')}}
function toggleM(){toggleM2()}
async function loadDevices(){try{const devs=await navigator.mediaDevices.enumerateDevices();const cams=devs.filter(d=>d.kind==='videoinput');const mics=devs.filter(d=>d.kind==='audioinput');$('camSelect').innerHTML='<option value="">📷 选择摄像头</option>'+cams.map((d,i)=>`<option value="${d.deviceId}">${d.label||'摄像头'+(i+1)}</option>`).join('');$('micSelect').innerHTML='<option value="">🎤 选择麦克风</option>'+mics.map((d,i)=>`<option value="${d.deviceId}">${d.label||'麦克风'+(i+1)}</option>`).join('')}catch(e){}}
window.onload=()=>{
    updateNpcVoiceButton();
    // 初始化场景选择，确保压力敏感区正确显示
    renderScenes();
    genMems();
    // 找到并选中默认场景，确保压力敏感区和酒局等级正确显示
    setTimeout(() => {
        const pressureSectionWrapper = $('pressureSectionWrapper');
        const banquetLevelWrapper = $('banquetLevelWrapper');
        
        if(scene.includes('家庭')){
            pressureSectionWrapper.style.display = 'block';
        }
        
        if(scene === '商务饭局谈判'){
            banquetLevelWrapper.style.display = 'block';
            applySceneInfo(banquetLevelDescriptions[selectedBanquetLevel]);
        }
    }, 100);
    loadDevices();
};

