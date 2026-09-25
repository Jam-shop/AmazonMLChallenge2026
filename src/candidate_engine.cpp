// Scalable blocking and pair-feature engine for Amazon ML Challenge 2026.
// Uses only challenge data and the C++17 standard library.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <atomic>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using std::string;
using u64 = uint64_t;
using u32 = uint32_t;

static u64 mix64(u64 x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static u64 hash_text(const string &s) {
    u64 h = 1469598103934665603ULL;
    for (unsigned char c : s) { h ^= c; h *= 1099511628211ULL; }
    return h;
}

static u64 entity_num(const string &s) {
    auto p = s.find('-');
    return std::stoull(s.substr(p == string::npos ? 0 : p + 1));
}

static std::array<string, 4> parse_row(const string &line) {
    std::array<string, 4> out;
    size_t a = 0;
    for (int i = 0; i < 3; ++i) {
        size_t b = line.find('\t', a);
        if (b == string::npos) { out[i] = line.substr(a); return out; }
        out[i] = line.substr(a, b - a); a = b + 1;
    }
    out[3] = line.substr(a);
    if (!out[3].empty() && out[3].back() == '\r') out[3].pop_back();
    return out;
}

static bool is_legal(const string &x) {
    static const std::unordered_set<string> words = {
        "inc","incorporated","corp","corporation","company","co","llc","limited","ltd",
        "private","pvt","plc","llp","sarl","sas","sa"
    };
    return words.count(x);
}

static string address_abbrev(const string &x) {
    static const std::unordered_map<string,string> m = {
        {"road","rd"},{"street","st"},{"avenue","ave"},{"boulevard","blvd"},
        {"drive","dr"},{"lane","ln"},{"highway","hwy"},{"suite","ste"},
        {"apartment","apt"},{"building","bldg"},{"floor","fl"}
    };
    auto it=m.find(x); return it==m.end()?x:it->second;
}

// Conservative UTF-8 normalization: ASCII case/punctuation plus common Latin accents.
static string normalize(const string &s, bool name) {
    string mapped; mapped.reserve(s.size()+4);
    for (size_t i=0;i<s.size();) {
        unsigned char c=s[i];
        if (c<128) {
            if (c>='A'&&c<='Z') c=(unsigned char)(c+32);
            if ((c>='a'&&c<='z')||(c>='0'&&c<='9')) mapped.push_back((char)c);
            else if (c=='&') mapped += " and "; else mapped.push_back(' ');
            ++i; continue;
        }
        // Decode enough Unicode to fold Latin-1 accents; retain other letters verbatim.
        u32 cp=0; size_t n=1;
        if ((c&0xE0)==0xC0 && i+1<s.size()) { cp=((c&31)<<6)|(s[i+1]&63); n=2; }
        else if ((c&0xF0)==0xE0 && i+2<s.size()) { cp=((c&15)<<12)|((s[i+1]&63)<<6)|(s[i+2]&63); n=3; }
        else if ((c&0xF8)==0xF0 && i+3<s.size()) { cp=((c&7)<<18)|((s[i+1]&63)<<12)|((s[i+2]&63)<<6)|(s[i+3]&63); n=4; }
        string repl;
        if ((cp>=0x300&&cp<=0x36f)) repl="";
        else if (string("ÀÁÂÃÄÅàáâãäå").find(s.substr(i,n))!=string::npos) repl="a";
        else if (string("Çç").find(s.substr(i,n))!=string::npos) repl="c";
        else if (string("ÈÉÊËèéêë").find(s.substr(i,n))!=string::npos) repl="e";
        else if (string("ÌÍÎÏìíîï").find(s.substr(i,n))!=string::npos) repl="i";
        else if (string("Ññ").find(s.substr(i,n))!=string::npos) repl="n";
        else if (string("ÒÓÔÕÖØòóôõöø").find(s.substr(i,n))!=string::npos) repl="o";
        else if (string("ÙÚÛÜùúûü").find(s.substr(i,n))!=string::npos) repl="u";
        else if (string("ÝŸýÿ").find(s.substr(i,n))!=string::npos) repl="y";
        else if (cp==0x2019||cp==0x2018||cp==0x2013||cp==0x2014||cp==0x00ab||cp==0x00bb) repl=" ";
        else repl=s.substr(i,n);
        mapped += repl; i += n;
    }
    std::istringstream in(mapped); string tok, out;
    while (in>>tok) {
        if (name && is_legal(tok)) continue;
        if (!name) tok=address_abbrev(tok);
        if (!out.empty()) out.push_back(' '); out += tok;
    }
    return out;
}

static std::vector<string> tokens(const string &s) {
    std::istringstream in(s); std::vector<string> v; string x;
    while (in>>x) v.push_back(x);
    std::sort(v.begin(),v.end()); v.erase(std::unique(v.begin(),v.end()),v.end()); return v;
}

static std::vector<u64> token_hashes(const string &s) {
    auto t=tokens(s); std::vector<u64> v; v.reserve(t.size());
    for(auto &x:t) v.push_back(hash_text(x)); return v;
}

static std::vector<u64> qgram_hashes(const string &s) {
    string x="^^"+s+"$$"; std::vector<u64> v;
    if (s.empty()) return v;
    for(size_t i=0;i+3<=x.size();++i){u64 h=1469598103934665603ULL;for(size_t k=i;k<i+3;++k){h^=(unsigned char)x[k];h*=1099511628211ULL;}v.push_back(h);}
    std::sort(v.begin(),v.end()); v.erase(std::unique(v.begin(),v.end()),v.end()); return v;
}

// Thirty-two independent 3-row MinHash bands. Requiring three band collisions makes
// random pairs vanishingly rare while retaining pairs with moderate overlap.
static std::vector<u64> lsh_keys(const string &s) {
    auto q=qgram_hashes(s); if(q.empty()) return {};
    std::array<u64,96> mins; mins.fill(std::numeric_limits<u64>::max());
    for(u64 h:q) for(int j=0;j<96;++j){
        u64 z=h*(0x9e3779b97f4a7c15ULL+2ULL*j)+(0xbf58476d1ce4e5b9ULL*(j+1));
        z^=z>>29; if(z<mins[j])mins[j]=z;
    }
    std::vector<u64> out;out.reserve(32);
    for(int b=0;b<32;++b)out.push_back(mix64(mins[3*b]^mix64(mins[3*b+1])^mix64(mins[3*b+2])^u64(b)));
    return out;
}

static std::vector<u64> compound_pairs(const std::vector<u64>&v){
    std::vector<u64> out;out.reserve(v.size()*(v.size()-1)/2);
    for(size_t i=0;i<v.size();++i)for(size_t j=i+1;j<v.size();++j){u64 a=v[i],b=v[j];if(a>b)std::swap(a,b);out.push_back(mix64(a^mix64(b)));}
    return out;
}

static std::vector<string> numeric_tokens(const string &s) {
    std::vector<string> v; string x;
    for(char c:s) { if(c>='0'&&c<='9') x.push_back(c); else if(!x.empty()){v.push_back(x);x.clear();} }
    if(!x.empty())v.push_back(x); std::sort(v.begin(),v.end()); v.erase(std::unique(v.begin(),v.end()),v.end()); return v;
}

struct Rec { u32 id; u64 country; string name, addr; };
struct Entry { u64 key; u32 idx; bool operator<(const Entry&o)const{return key<o.key||(key==o.key&&idx<o.idx);} };

static u64 make_key(int tag,u64 country,u64 val){return mix64(val ^ mix64(country) ^ (u64(tag)*0x9e3779b97f4a7c15ULL));}

static double set_jaccard(const std::vector<string>&a,const std::vector<string>&b){
    size_t i=0,j=0,inter=0,uni=0; while(i<a.size()||j<b.size()){
        if(j==b.size()||(i<a.size()&&a[i]<b[j])){++i;++uni;}
        else if(i==a.size()||b[j]<a[i]){++j;++uni;} else {++i;++j;++inter;++uni;}
    } return uni?double(inter)/uni:1.0;
}
static double qdice(const string&a,const string&b){
    auto x=qgram_hashes(a),y=qgram_hashes(b); size_t i=0,j=0,k=0;
    while(i<x.size()&&j<y.size()){if(x[i]<y[j])++i;else if(y[j]<x[i])++j;else{++i;++j;++k;}}
    return x.size()+y.size()?2.0*k/(x.size()+y.size()):1.0;
}
static double edit_sim(const string&a,const string&b){
    if(a.empty()||b.empty())return a==b?1.0:0.0;
    const string *x=&a,*y=&b;if(x->size()>y->size())std::swap(x,y);
    std::vector<int> p(x->size()+1),c(x->size()+1);std::iota(p.begin(),p.end(),0);
    for(size_t j=1;j<=y->size();++j){c[0]=j;for(size_t i=1;i<=x->size();++i)c[i]=std::min({c[i-1]+1,p[i]+1,p[i-1]+((*x)[i-1]!=(*y)[j-1])});p.swap(c);}
    return 1.0-double(p[x->size()])/std::max(a.size(),b.size());
}
static double length_ratio(const string&a,const string&b){return a.empty()||b.empty()?0.0:double(std::min(a.size(),b.size()))/std::max(a.size(),b.size());}
static double containment(const string&a,const string&b){if(a.empty()||b.empty())return 0;return (a.find(b)!=string::npos||b.find(a)!=string::npos)?1.0:0.0;}

static std::vector<double> features(const Rec&a,const Rec&b,int source,int hits){
    auto an=tokens(a.name),bn=tokens(b.name),aa=tokens(a.addr),ba=tokens(b.addr);
    double name_dice=qdice(a.name,b.name), addr_dice=qdice(a.addr,b.addr);
    auto ax=numeric_tokens(a.addr),bx=numeric_tokens(b.addr);
    bool any_num=!ax.empty()&&!bx.empty(); size_t ni=0,i=0,j=0;
    while(i<ax.size()&&j<bx.size()){if(ax[i]<bx[j])++i;else if(bx[j]<ax[i])++j;else{++ni;++i;++j;}}
    auto postal=[](const std::vector<string>&v){std::vector<string>z;for(auto&s:v)if(s.size()==5||s.size()==6)z.push_back(s);return z;};
    auto ap=postal(ax),bp=postal(bx); size_t pi=0; i=j=0;
    while(i<ap.size()&&j<bp.size()){if(ap[i]<bp[j])++i;else if(bp[j]<ap[i])++j;else{++pi;++i;++j;}}
    return {
        double(source==3), double(a.name==b.name&&!a.name.empty()), double(a.addr==b.addr&&!a.addr.empty()),
        set_jaccard(an,bn), name_dice, edit_sim(a.name,b.name), containment(a.name,b.name), length_ratio(a.name,b.name),
        set_jaccard(aa,ba), addr_dice, edit_sim(a.addr,b.addr), containment(a.addr,b.addr), length_ratio(a.addr,b.addr),
        double(ni>0), double(any_num&&ni==0), any_num?double(ni)/std::max(ax.size(),bx.size()):0.0,
        double(pi>0), double(!ap.empty()&&!bp.empty()&&pi==0), std::min(hits,10)/10.0,
        std::max(name_dice,addr_dice), std::min(name_dice,addr_dice)
    };
}

static std::vector<u64> choose_rare(const std::vector<u64>&vals,const std::unordered_map<u64,u32>&df,int k,u32 maxdf){
    std::vector<std::pair<u32,u64>> x; x.reserve(vals.size());
    for(u64 h:vals){auto it=df.find(h);if(it!=df.end()&&it->second<=maxdf)x.push_back({it->second,h});}
    std::sort(x.begin(),x.end()); if((int)x.size()>k)x.resize(k);
    std::vector<u64> out;for(auto&p:x)out.push_back(p.second);return out;
}

struct Args {
    string data="dataset/train", features_out="analysis/candidate_features.tsv", candidates_out="", matching_out="", model="";
    int sample_mod=10,sample_rem=0, max_df=80, max_bucket=300, top_k=100;
};
static Args parse_args(int argc,char**argv){Args a;for(int i=1;i<argc;++i){string k=argv[i];auto val=[&](){if(++i>=argc)throw std::runtime_error("missing value");return string(argv[i]);};
    if(k=="--data")a.data=val();else if(k=="--features-out")a.features_out=val();else if(k=="--candidates-out")a.candidates_out=val();else if(k=="--matching-out")a.matching_out=val();else if(k=="--model")a.model=val();else if(k=="--sample-mod")a.sample_mod=std::stoi(val());else if(k=="--sample-rem")a.sample_rem=std::stoi(val());else if(k=="--max-df")a.max_df=std::stoi(val());else if(k=="--max-bucket")a.max_bucket=std::stoi(val());else if(k=="--top-k")a.top_k=std::stoi(val());else throw std::runtime_error("unknown arg "+k);}return a;}

struct TreeNode {int feature=0,left=0,right=0;double threshold=0,value=0;bool leaf=false,missing_left=false;};
struct Model { string type="logistic";double intercept=0, threshold=1e100; std::vector<double> coef;std::vector<std::vector<TreeNode>> trees; };
static Model load_model(const string&p){Model m;if(p.empty())return m;std::ifstream f(p);string key;while(f>>key){
    if(key=="type")f>>m.type;else if(key=="intercept"||key=="baseline")f>>m.intercept;else if(key=="threshold")f>>m.threshold;
    else if(key=="coefficients"){string rest;std::getline(f,rest);std::istringstream in(rest);double x;while(in>>x)m.coef.push_back(x);}
    else if(key=="trees"){int n;f>>n;m.trees.reserve(n);}
    else if(key=="tree"){int n;f>>n;std::vector<TreeNode>t;t.reserve(n);for(int i=0;i<n;++i){string node;TreeNode x;int leaf,miss;f>>node>>x.feature>>x.threshold>>x.left>>x.right>>x.value>>leaf>>miss;x.leaf=leaf;x.missing_left=miss;t.push_back(x);}m.trees.push_back(std::move(t));}
}return m;}
static double score_model(const Model&m,const std::vector<double>&x){double score=m.intercept;if(m.type=="hgb"){for(auto&t:m.trees){int i=0;while(!t[i].leaf)i=(std::isnan(x[t[i].feature])?t[i].missing_left:x[t[i].feature]<=t[i].threshold)?t[i].left:t[i].right;score+=t[i].value;}}else for(size_t j=0;j<x.size()&&j<m.coef.size();++j)score+=m.coef[j]*x[j];return score;}

struct RankedCandidate {u64 target; float rank; uint16_t hits;};
static bool heap_cmp(const RankedCandidate&a,const RankedCandidate&b){return a.rank>b.rank||(a.rank==b.rank&&a.target>b.target);} // min-heap
struct Selected {u64 target;u32 idx;uint16_t hits;};

int main(int argc,char**argv){
 try{
    Args args=parse_args(argc,argv); auto started=std::chrono::steady_clock::now();
    std::vector<Rec>s1; std::unordered_map<u32,u32>s1_lookup;
    std::unordered_map<u64,u32> ntdf,atdf,nqdf,aqdf;
    std::ifstream f(args.data+"/"+(args.data.find("test")!=string::npos?"test":"train")+"_source1.tsv");
    if(!f)throw std::runtime_error("cannot open source1");string line;std::getline(f,line);
    while(std::getline(f,line)){auto r=parse_row(line);u32 id=(u32)entity_num(r[0]);if(args.sample_mod>1&&(int)(id%args.sample_mod)!=args.sample_rem)continue;
        Rec x{id,hash_text(r[3]),normalize(r[1],true),normalize(r[2],false)};u32 idx=s1.size();s1_lookup[id]=idx;s1.push_back(std::move(x));
        auto add=[](auto&map,const auto&v){for(u64 h:v)++map[h];}; add(ntdf,token_hashes(s1.back().name));add(atdf,token_hashes(s1.back().addr));add(nqdf,qgram_hashes(s1.back().name));add(aqdf,qgram_hashes(s1.back().addr));
    }
    std::cerr<<"loaded_s1="<<s1.size()<<" df_sizes="<<ntdf.size()<<","<<atdf.size()<<","<<nqdf.size()<<","<<aqdf.size()<<"\n";
    std::vector<Entry> index;index.reserve(s1.size()*75);
    auto addkey=[&](u64 key,u32 idx){index.push_back({key,idx});};
    for(u32 idx=0;idx<s1.size();++idx){auto&r=s1[idx];if(!r.name.empty())addkey(make_key(1,r.country,hash_text(r.name)),idx);if(!r.addr.empty())addkey(make_key(2,r.country,hash_text(r.addr)),idx);
        for(u64 h:choose_rare(token_hashes(r.name),ntdf,3,args.max_df))addkey(make_key(3,r.country,h),idx);
        for(u64 h:choose_rare(token_hashes(r.addr),atdf,3,args.max_df))addkey(make_key(4,r.country,h),idx);
        u32 indiv_limit=std::max(1,args.max_df/100);
        for(u64 h:choose_rare(qgram_hashes(r.name),nqdf,5,indiv_limit))addkey(make_key(5,r.country,h),idx);
        for(u64 h:choose_rare(qgram_hashes(r.addr),aqdf,5,indiv_limit))addkey(make_key(6,r.country,h),idx);
        for(u64 h:compound_pairs(choose_rare(qgram_hashes(r.name),nqdf,8,args.max_df)))addkey(make_key(7,r.country,h),idx);
        for(u64 h:compound_pairs(choose_rare(qgram_hashes(r.addr),aqdf,8,args.max_df)))addkey(make_key(8,r.country,h),idx);
    }
    std::sort(index.begin(),index.end());std::cerr<<"index_entries="<<index.size()<<"\n";
    // True ownership exists only for train mode and selected S1 rows.
    std::unordered_map<u64,u32>true_owner;std::vector<u32>true_counts(s1.size(),0),found_counts(s1.size(),0);
    string gt=args.data+"/train_ground_truth.tsv";std::ifstream g(gt);
    if(g){std::getline(g,line);while(std::getline(g,line)){size_t t=line.find('\t');string sid=line.substr(0,t);u32 id=(u32)entity_num(sid);auto it=s1_lookup.find(id);if(it==s1_lookup.end()||t==string::npos)continue;string rest=line.substr(t+1);std::stringstream ss(rest);string mid;while(std::getline(ss,mid,',')){if(mid.empty())continue;int src=mid[1]-'0';u32 n=(u32)entity_num(mid);true_owner[(u64(src)<<32)|n]=it->second;++true_counts[it->second];}}}
    Model model=load_model(args.model); bool final_mode=!args.candidates_out.empty();
    std::vector<std::vector<RankedCandidate>> top(s1.size());
    std::atomic<u64> pre_pairs{0},pre_true{0};std::atomic<size_t> max_candidates_target{0};std::array<std::mutex,4096> heap_locks;
    for(int source=2;source<=3;++source){string split=args.data.find("test")!=string::npos?"test":"train";std::ifstream in(args.data+"/"+split+"_source"+std::to_string(source)+".tsv");if(!in)throw std::runtime_error("cannot open target source");std::getline(in,line);u64 seen=0;std::vector<string> batch;batch.reserve(10000);
      while(in){batch.clear();for(int z=0;z<10000&&std::getline(in,line);++z)batch.push_back(line);
        #pragma omp parallel for schedule(dynamic,64)
        for(size_t bi=0;bi<batch.size();++bi){auto rr=parse_row(batch[bi]);Rec b{(u32)entity_num(rr[0]),hash_text(rr[3]),normalize(rr[1],true),normalize(rr[2],false)};
            std::vector<std::pair<int,u64>> keys;if(!b.name.empty())keys.push_back({1,make_key(1,b.country,hash_text(b.name))});if(!b.addr.empty())keys.push_back({2,make_key(2,b.country,hash_text(b.addr))});
            for(u64 h:choose_rare(token_hashes(b.name),ntdf,5,args.max_df))keys.push_back({3,make_key(3,b.country,h)});for(u64 h:choose_rare(token_hashes(b.addr),atdf,5,args.max_df))keys.push_back({4,make_key(4,b.country,h)});
            u32 indiv_limit=std::max(1,args.max_df/100);
            for(u64 h:choose_rare(qgram_hashes(b.name),nqdf,9,indiv_limit))keys.push_back({5,make_key(5,b.country,h)});for(u64 h:choose_rare(qgram_hashes(b.addr),aqdf,9,indiv_limit))keys.push_back({6,make_key(6,b.country,h)});
            for(u64 h:compound_pairs(choose_rare(qgram_hashes(b.name),nqdf,8,args.max_df)))keys.push_back({7,make_key(7,b.country,h)});for(u64 h:compound_pairs(choose_rare(qgram_hashes(b.addr),aqdf,8,args.max_df)))keys.push_back({8,make_key(8,b.country,h)});
            // Packed evidence counters: exact bits, then 4-bit token/qgram hit counts.
            std::unordered_map<u32,u32> cand;
            for(auto [tag,key]:keys){Entry lo{key,0},hi{key,std::numeric_limits<u32>::max()};auto a=std::lower_bound(index.begin(),index.end(),lo),z=std::upper_bound(index.begin(),index.end(),hi);if(z-a>args.max_bucket)continue;for(;a!=z;++a){u32 &v=cand[a->idx];if(tag==1)v|=1;else if(tag==2)v|=2;else {int shift=4+(tag-3)*4;u32 n=(v>>shift)&15;if(n<15)v+=1u<<shift;}}}
            size_t oldmax=max_candidates_target.load(std::memory_order_relaxed);while(cand.size()>oldmax&&!max_candidates_target.compare_exchange_weak(oldmax,cand.size(),std::memory_order_relaxed)){}u64 target_key=(u64(source)<<32)|b.id;
            for(auto [idx,evidence]:cand){int nt=(evidence>>4)&15,at=(evidence>>8)&15,nq=(evidence>>12)&15,aq=(evidence>>16)&15,ln=(evidence>>20)&15,la=(evidence>>24)&15;bool exact=evidence&3;
                bool keep=exact||ln>=1||la>=1||(nt>=1&&nq>=1)||(at>=1&&aq>=1)||nq>=3||aq>=4;if(!keep)continue;int hits=nt+at+nq+aq+3*(ln+la)+5*int(exact);
                double block_name_dice=qdice(s1[idx].name,b.name),block_addr_dice=qdice(s1[idx].addr,b.addr);
                if(std::max(block_name_dice,block_addr_dice)<0.40)continue;
                pre_pairs.fetch_add(1,std::memory_order_relaxed);auto owner=true_owner.find(target_key);if(owner!=true_owner.end()&&owner->second==idx)pre_true.fetch_add(1,std::memory_order_relaxed);
                auto nx=numeric_tokens(s1[idx].addr),ny=numeric_tokens(b.addr);size_t xi=0,yi=0,common_num=0;while(xi<nx.size()&&yi<ny.size()){if(nx[xi]<ny[yi])++xi;else if(ny[yi]<nx[xi])++yi;else{++common_num;++xi;++yi;}}
                bool num_conflict=!nx.empty()&&!ny.empty()&&!common_num;float rank=float(std::max(block_name_dice,block_addr_dice)+0.70*std::min(block_name_dice,block_addr_dice)+0.08*int(common_num>0)-0.12*int(num_conflict)+0.05*int((evidence&3)==3)+0.002*std::min(hits,10));
                RankedCandidate rc{target_key,rank,(uint16_t)hits};std::lock_guard<std::mutex> guard(heap_locks[idx%heap_locks.size()]);auto &heap=top[idx];
                if((int)heap.size()<args.top_k){heap.push_back(rc);std::push_heap(heap.begin(),heap.end(),heap_cmp);}
                else if(heap_cmp(rc,heap.front())){std::pop_heap(heap.begin(),heap.end(),heap_cmp);heap.back()=rc;std::push_heap(heap.begin(),heap.end(),heap_cmp);}
            }
        }
        seen+=batch.size();if(seen%100000==0)std::cerr<<"source="<<source<<" seen="<<seen<<" pre_pairs="<<pre_pairs.load()<<"\n";
      }
    }
    std::vector<Selected> selected;selected.reserve(s1.size()*std::min(args.top_k,50));
    for(u32 idx=0;idx<top.size();++idx)for(auto &r:top[idx])selected.push_back({r.target,idx,r.hits});
    std::sort(selected.begin(),selected.end(),[](const Selected&a,const Selected&b){return a.target<b.target||(a.target==b.target&&a.idx<b.idx);});
    u64 positive_pairs=0;for(auto&s:selected){auto it=true_owner.find(s.target);if(it!=true_owner.end()&&it->second==s.idx){++positive_pairs;++found_counts[s.idx];}}
    u64 pairs=selected.size(),total_true=std::accumulate(true_counts.begin(),true_counts.end(),u64(0)),found_true=std::accumulate(found_counts.begin(),found_counts.end(),u64(0));
    std::cerr<<"pre_pairs="<<pre_pairs.load()<<" pre_recall="<<(total_true?double(pre_true.load())/total_true:0)<<" final_pairs="<<pairs<<" positives_found="<<positive_pairs<<" true_total="<<total_true<<" candidate_recall="<<(total_true?double(found_true)/total_true:0)<<" avg_candidates="<<double(pairs)/s1.size()<<" max_raw_per_target="<<max_candidates_target.load()<<"\n";
    std::ofstream feat;if(!args.features_out.empty()){feat.open(args.features_out);feat<<"s1_id\ttarget_id\tfold\tlabel\tsource\tname_exact\taddress_exact\tname_token_jaccard\tname_qgram_dice\tname_edit\tname_containment\tname_length_ratio\taddress_token_jaccard\taddress_qgram_dice\taddress_edit\taddress_containment\taddress_length_ratio\tnumeric_agree\tnumeric_conflict\tnumeric_overlap_ratio\tpostal_agree\tpostal_conflict\tretrieval_hits\tbest_field\tworst_field\n";feat<<std::setprecision(7);}
    std::vector<std::vector<u64>> matched_ids(final_mode?s1.size():0);
    if(feat||final_mode){
      for(int source=2;source<=3;++source){string split=args.data.find("test")!=string::npos?"test":"train";std::ifstream in(args.data+"/"+split+"_source"+std::to_string(source)+".tsv");std::getline(in,line);
        while(std::getline(in,line)){auto rr=parse_row(line);u32 tid=(u32)entity_num(rr[0]);u64 target_key=(u64(source)<<32)|tid;Selected needle{target_key,0,0};auto a=std::lower_bound(selected.begin(),selected.end(),needle,[](const Selected&x,const Selected&y){return x.target<y.target;});if(a==selected.end()||a->target!=target_key)continue;Rec b{tid,hash_text(rr[3]),normalize(rr[1],true),normalize(rr[2],false)};
          for(;a!=selected.end()&&a->target==target_key;++a){auto x=features(s1[a->idx],b,source,a->hits);bool label=false;auto ti=true_owner.find(target_key);if(ti!=true_owner.end()&&ti->second==a->idx)label=true;
            if(feat){feat<<"S1-"<<s1[a->idx].id<<'\t'<<'S'<<source<<'-'<<tid<<'\t'<<(mix64(s1[a->idx].id)%5==0)<<'\t'<<label;for(double v:x)feat<<'\t'<<v;feat<<'\n';}
            if(final_mode){double score=score_model(model,x);if(score>=model.threshold)matched_ids[a->idx].push_back(target_key);}
          }
        }
      }
    }
    if(final_mode){
        std::ofstream co(args.candidates_out);co<<"source1_entity_id\tcandidate_entity_ids\n";for(size_t i=0;i<s1.size();++i){co<<"S1-"<<s1[i].id<<'\t';std::vector<u64>v;for(auto&r:top[i])v.push_back(r.target);std::sort(v.begin(),v.end());for(size_t j=0;j<v.size();++j){if(j)co<<',';co<<'S'<<(v[j]>>32)<<'-'<<(u32)v[j];}co<<'\n';}
        std::ofstream mo(args.matching_out);mo<<"source1_entity_id\tmatched_entity_ids\n";for(size_t i=0;i<s1.size();++i){mo<<"S1-"<<s1[i].id<<'\t';auto &v=matched_ids[i];std::sort(v.begin(),v.end());v.erase(std::unique(v.begin(),v.end()),v.end());for(size_t j=0;j<v.size();++j){if(j)mo<<',';mo<<'S'<<(v[j]>>32)<<'-'<<(u32)v[j];}mo<<'\n';}
    }
    double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();std::cerr<<"runtime_seconds="<<sec<<"\n";
 }catch(const std::exception&e){std::cerr<<"ERROR: "<<e.what()<<"\n";return 1;}return 0;
}
