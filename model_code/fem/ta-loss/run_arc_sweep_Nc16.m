function run_arc_sweep_Nc16()
  % ARC 16-饼(主配置 arc_16pancake_nuc600) T-A 全扫: 3温度×6Npw @rho=5000 + Npw20@20K×rho{50,100,1500}.
  % Np=16; Nt 取50整倍(满足三段式 Nt3=Nt-200 被50整除); Ip 按安匝8.4MA反算. 断点续跑.
  % 输出到 results/Nc16/, 不覆盖 Nc12 旧数据. 导出 [t, int4(mag), gev9(radial)].
  addpath(fullfile(getenv('COMSOL_ROOT'),'mli')); mphstart(2036);
  import com.comsol.model.util.*;
  base=[fullfile(getenv('HTSMAG_FEM_DATA_ROOT'),'2-TA-loss') filesep];
  res=[fullfile(getenv('HTSMAG_RESULTS_ROOT'),'Nc16') filesep];
  if ~exist(res,'dir'); mkdir(res); end
  diary([res 'sweep_Nc16.log']);
  fprintf('=== ARC Nc=16 全扫 START %s ===\n', datestr(now,31));
  % 温度 -> [Nt(50整倍), Ip=8.4e6/(Nt*16)]
  Tlist=[4.2 10 20];
  Nt_of =containers.Map([4.2 10 20],[700 850 1150]);
  Ip_of =containers.Map([4.2 10 20],[750 617.6 456.5]);
  Npw_list=[5 10 20 50 100 200]; rho_main=5000;
  m=mphload([base 'ARC_TA_loss_Nc12_R2.mph']); fprintf('LOADED (改 Np 12->16)\n');
  m.param.set('Np','16');   % 16 饼
  m.study('std1').feature('param').active(false); m.study('std1').feature('param1').active(false);
  m.study('std1').feature('time').set('tlist','range(0,td/16,2*td)');
  t1=m.sol('sol1').feature('t1');
  t1.set('tlist','range(0,td/16,2*td)'); t1.set('rtol','1e-4'); t1.set('initialstepbdfactive',false);
  t1.set('maxstepbdfactive',true); t1.set('maxstepbdf','4000');
  m.mesh('mesh1').feature('dis1').set('numelem',20); m.mesh('mesh1').feature('size2').set('hmax',0.005);

  function do_solve(csv, Npw, rho)
    if exist(csv,'file'); fprintf('SKIP %s (已有)\n', csv); return; end
    m.param('par2').set('Npw',num2str(Npw));
    m.param('par7').set('rho_turn',[num2str(rho) '[uohm*cm^2]']);
    fprintf('SOLVING Npw=%d rho=%d ... %s\n',Npw,rho,datestr(now,31)); ts=tic;
    try
      m.sol('sol1').runAll;
      try; m.result.table.create('tmag','Table'); catch; end
      m.result.numerical('int4').set('table','tmag'); m.result.numerical('int4').set('data','dset1'); m.result.numerical('int4').setResult; Dm=mphtable(m,'tmag').data;
      try; m.result.table.create('trad','Table'); catch; end
      m.result.numerical('gev9').set('table','trad'); m.result.numerical('gev9').set('data','dset1'); m.result.numerical('gev9').setResult; Dr=mphtable(m,'trad').data;
      writematrix([Dm(:,1) Dm(:,2) Dr(:,2)], csv);
      fprintf('   OK %.1fmin  磁化peak=%.2f W, 径向peak=%.1f W\n', toc(ts)/60, max(Dm(:,2)), max(Dr(:,2)));
    catch ME
      fprintf(2,'   FAIL Npw=%d rho=%d: %s\n',Npw,rho,ME.message);
    end
  end

  for T=Tlist
    m.param('par6').set('T',[num2str(T) '[K]']); m.param('par6').set('Nt',num2str(Nt_of(T))); m.param('par6').set('Ip',num2str(Ip_of(T)));
    fprintf('\n--- T=%gK: Np=16, Nt=%d, Ip=%g, 重建几何... %s ---\n', T, Nt_of(T), Ip_of(T), datestr(now,31));
    try
      m.geom('geom1').run;
      e=1e-3;
      pmc=mphselectbox(m,'geom1',[3-e 6+e; -e e],'boundary');
      mi=unique([mphselectbox(m,'geom1',[3-e 3+e; -e 1.2+e],'boundary') mphselectbox(m,'geom1',[6-e 6+e; -e 1.2+e],'boundary') mphselectbox(m,'geom1',[3-e 6+e; 1.2-e 1.2+e],'boundary')]);
      m.physics('mf').feature('pmc1').selection.set(pmc); m.physics('mf').feature('mi2').selection.set(mi);
      m.mesh('mesh1').run; fprintf('  GEOM+MESH OK elems=%d\n', m.mesh('mesh1').getNumElem);
    catch ME
      fprintf(2,'  GEOM FAIL T=%g: %s (跳过该温度)\n',T,ME.message); continue;
    end
    for Npw=Npw_list
      do_solve([res sprintf('ARC16_sweep_T%g_Npw%d.csv',T,Npw)], Npw, rho_main);
    end
    if T==20   % Fig2 面板c: 20K/Npw20 补 rho{50,100,1500}
      for rho=[50 100 1500]
        do_solve([res sprintf('ARC16_sweep_T20_Npw20_rho%d.csv',rho)], 20, rho);
      end
    end
  end
  fprintf('\n=== ARC Nc=16 全扫 DONE %s ===\nNC16_SWEEP_COMPLETE\n', datestr(now,31));
  diary off;
end
