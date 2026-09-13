function ipT_resolve()
  addpath(fullfile(getenv('COMSOL_ROOT'),'mli')); mphstart(2036);
  import com.comsol.model.util.*;
  base=[fullfile(getenv('HTSMAG_FEM_DATA_ROOT'),'1-field-FEM') filesep];
  res=[getenv('HTSMAG_RESULTS_ROOT') filesep];
  diary([res 'ipT_resolve.log']);
  try
    m=mphload([base 'ARC_field_mangiarotti.mph']);
    Nc=12; AT=8.4e6; lf=0.7;
    fprintf('=== 权威 Ip(T)/Nt(T) @载流比%.2f (COMSOL体积min, 各温度重解刷新快照) ===\n', lf);
    fprintf(' T[K]  minJc[A/mm2]  Ic_tape[A]  Ip(0.7)[A]  Ip取整  Nt    lf@Ip400\n');
    M=[];
    for T=[4.2 10 20]
      m.param.set('Top', num2str(T));
      m.sol('sol1').runAll;   % 同场, 仅刷新 Top 快照
      Jcmin = mphmin(m,'Jc_mang','volume','selection',2,'dataset','dset1');
      Ic=1.2*Jcmin; Ip=lf*Ic; Ip_r=round(Ip/10)*10; Nt=AT/(Nc*Ip_r); lf400=400/Ic;
      fprintf(' %4.1f  %8.0f     %7.0f    %7.0f    %5d  %5.0f   %.3f\n', T,Jcmin,Ic,Ip,Ip_r,Nt,lf400);
      M=[M; T Jcmin Ic Ip Ip_r Nt lf400];
    end
    writematrix(M,[res 'ip_vs_T_comsol.csv']);
    m.param.set('Top','20'); m.sol('sol1').runAll;
    mphsave(m,[base 'ARC_field_mangiarotti.mph']);
    fprintf('SAVED (Top=20). IPT_RESOLVE_OK\n');
  catch ME
    fprintf(2,'ERR: %s\n', getReport(ME,'basic','hyperlinks','off'));
  end
  diary off;
end
