#define main candidate_engine_original_main
#include "candidate_engine.cpp"
#undef main

// Materialize features for every labeled positive pair.  This is used only for
// the final all-training-data fit; hard negatives come from candidate_features.tsv.
int main(int argc,char**argv){
  if(argc!=3){std::cerr<<"usage: full_positive_features DATASET_TRAIN OUTPUT\n";return 2;}
  string data=argv[1],outpath=argv[2],line;std::vector<Rec>s1;std::unordered_map<u32,u32>lookup;
  std::ifstream f(data+"/train_source1.tsv");std::getline(f,line);
  while(std::getline(f,line)){auto r=parse_row(line);u32 id=(u32)entity_num(r[0]);lookup[id]=s1.size();s1.push_back({id,hash_text(r[3]),normalize(r[1],true),normalize(r[2],false)});}
  std::cerr<<"loaded_s1="<<s1.size()<<"\n";
  std::unordered_map<u64,u32>owner;owner.reserve(8000000);std::ifstream g(data+"/train_ground_truth.tsv");std::getline(g,line);
  while(std::getline(g,line)){size_t t=line.find('\t');u32 sid=(u32)entity_num(line.substr(0,t));auto si=lookup.find(sid);if(si==lookup.end())continue;std::stringstream ss(line.substr(t+1));string mid;while(std::getline(ss,mid,',')){if(mid.empty())continue;owner[(u64(mid[1]-'0')<<32)|(u32)entity_num(mid)]=si->second;}}
  std::cerr<<"positive_targets="<<owner.size()<<"\n";
  std::ofstream out(outpath);out<<"s1_id\ttarget_id\tfold\tlabel\tsource\tname_exact\taddress_exact\tname_token_jaccard\tname_qgram_dice\tname_edit\tname_containment\tname_length_ratio\taddress_token_jaccard\taddress_qgram_dice\taddress_edit\taddress_containment\taddress_length_ratio\tnumeric_agree\tnumeric_conflict\tnumeric_overlap_ratio\tpostal_agree\tpostal_conflict\tretrieval_hits\tbest_field\tworst_field\n";out<<std::setprecision(7);
  u64 written=0;
  for(int source=2;source<=3;++source){std::ifstream in(data+"/train_source"+std::to_string(source)+".tsv");std::getline(in,line);std::vector<string>batch;batch.reserve(10000);
    while(in){batch.clear();for(int i=0;i<10000&&std::getline(in,line);++i)batch.push_back(line);std::vector<string>rows(batch.size());
      #pragma omp parallel for schedule(dynamic,128)
      for(size_t i=0;i<batch.size();++i){auto r=parse_row(batch[i]);u32 tid=(u32)entity_num(r[0]);u64 key=(u64(source)<<32)|tid;auto it=owner.find(key);if(it==owner.end())continue;u32 idx=it->second;Rec b{tid,hash_text(r[3]),normalize(r[1],true),normalize(r[2],false)};auto x=features(s1[idx],b,source,5);std::ostringstream ss;ss<<std::setprecision(7)<<"S1-"<<s1[idx].id<<'\t'<<'S'<<source<<'-'<<tid<<"\t0\t1";for(double v:x)ss<<'\t'<<v;ss<<'\n';rows[i]=ss.str();}
      for(auto&r:rows)if(!r.empty()){out<<r;++written;}if(written&&written%1000000<10000)std::cerr<<"written="<<written<<"\n";
    }
  }
  std::cerr<<"all_positive_rows="<<written<<"\n";return written==owner.size()?0:1;
}
